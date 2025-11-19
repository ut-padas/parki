__all__ = [
    "create_buffers",
    "buffer_nbytes",
    "compute_slab",
    "distribute_points",
    "bucket_sort",
    "communicate_ghost_points",
    "communicate_grid_ghost_points",
    "gather_points",
]

import numpy as np

try:
    from mpi4py import MPI
except ModuleNotFoundError:
    MPI = None
from parkipy.utils import get_array_module


def create_buffers(mpi_comm, execution_space, size):
    """
    Create buffer arrays with room for ``size`` elements. The
    optional ``MPI.Comm`` communication context ``mpi_comm`` is
    used to determine the MPI size for some of the buffers.

    The buffers are returned in a dict object.
    """
    cp = get_array_module(execution_space)

    if mpi_comm is not None:
        mpi_size = mpi_comm.Get_size()
    else:
        return dict()

    send_buffer = cp.empty((size, 3), order="C")
    recv_buffer = cp.empty((size, 3), order="C")
    all_to_all_send_count_buffer = np.empty(mpi_size, dtype=np.int64)
    all_to_all_recv_count_buffer = np.empty(mpi_size, dtype=np.int64)

    buffers = {
        "send": send_buffer,
        "recv": recv_buffer,
        "all_to_all_send_count": all_to_all_send_count_buffer,
        "all_to_all_recv_count": all_to_all_recv_count_buffer,
    }
    return buffers


def buffer_nbytes(buffers):
    """
    Return the total number of bytes taken up by the buffers in
    the dict ``buffers`` (assumed to be created by ``create_buffers()``).
    """
    tot = 0
    for buffer in buffers.values():
        tot += buffer.nbytes
    return tot


def compute_slab(box, mpi_comm=None):
    """
    Given a box size ``box`` and an optional ``MPI.Comm``
    communication context ``mpi_comm``, compute a dict ``slab``
    with slab properties for the current rank. The keys in
    ``slab`` are:

        - "width": slab width
        - "left": left endpoint of slab
        - "right": right endpoint of slab
    """
    if mpi_comm is not None:
        mpi_size = mpi_comm.Get_size()
        mpi_rank = mpi_comm.Get_rank()
    else:
        mpi_size = 1
        mpi_rank = 0

    slab = dict()
    slab["width"] = box[0] / mpi_size
    slab["left"] = mpi_rank * slab["width"]
    slab["right"] = (mpi_rank + 1) * slab["width"]
    slab["slab box"] = [slab["width"], box[1], box[2]]
    return slab


def distribute_points(N, mpi_comm=None):
    """
    Given a total number of points ``N`` and an optional
    ``MPI.Comm`` communication context ``mpi_comm``, compute the
    number of points ``n`` that should go into the current rank,
    assuming uniform distribution of points among the rank.

    This function ensures that summing up ``n`` over all the
    ranks will yield ``N`` (even when ``N`` is not divisible by
    the MPI size).
    """
    if mpi_comm is not None:
        mpi_size = mpi_comm.Get_size()
        mpi_rank = mpi_comm.Get_rank()
    else:
        mpi_size = 1
        mpi_rank = 0

    # All ranks should get at least floor(N/mpi_size)
    n = N // mpi_size
    # Some should get one extra (for simplicity the first ranks)
    if mpi_rank < N % mpi_size:
        n += 1
    return n


def bucket_sort(
    mpi_comm,
    execution_space,
    box,
    trg,
    nt,
    src=None,
    ns=None,
    dens=None,
    normal=None,
    buffers=None,
):
    """
    Given a box size ``box`, arrays ``trg``, ``src``,
    ``dens_sl``, ``dens_dl``, ``normal``, integers ``nt``, ``ns``,
    and an optional ``MPI.Comm`` communication context ``mpi_comm``,
    perform bucket sorting of target and source points across the MPI
    ranks.

    ``buffers`` may be given and will be used for sending and receiving
    MPI messages. (It is a dict object as returned by ``create_buffers()``.)

    ``nt`` and ``ns`` are the local target and source counts,
    respectively.

    This function handles the communication between the rank.

    Returns a tuple ``(new_nt, new_ns)`` with new values for
    ``nt`` and ``ns``.
    """
    cp = get_array_module(execution_space)

    if mpi_comm is not None:
        mpi_size = mpi_comm.Get_size()
    else:
        return nt, ns

    slab_width = box[0] / mpi_size
    slabs = slab_width * cp.arange(mpi_size + 1)
    new_nt = _bucket_sort_core(
        mpi_comm, execution_space, slabs, trg, nt, buffers=buffers, base_tag=10
    )
    new_ns = None
    if src is not None:
        new_ns = _bucket_sort_core(
            mpi_comm,
            execution_space,
            slabs,
            src,
            ns,
            dens,
            normal,
            buffers=buffers,
            base_tag=20,
        )
    return new_nt, new_ns


def _bucket_sort_core(
    mpi_comm, execution_space, slabs, arr, count, *extra_arr, buffers=None, base_tag=0
):
    cp = get_array_module(execution_space)

    if mpi_comm is not None:
        mpi_size = mpi_comm.Get_size()
        mpi_rank = mpi_comm.Get_rank()
    else:
        return count

    if buffers is None:
        buffers = create_buffers(arr.shape[0], mpi_comm)

    # Send and receive points to/from others
    send_count = buffers["all_to_all_send_count"]
    recv_count = buffers["all_to_all_recv_count"]
    send_buffer = buffers["send"]
    recv_buffer = buffers["recv"]
    # First figure out what to send to the others
    view = arr[:count, 0] % slabs[-1]
    arr[:count, 0] = view  # store recentered coordinates
    offset = 0
    I_store = []
    for i in range(mpi_size):
        I = (view >= slabs[i]) & (view < slabs[i + 1])
        I_store.append(I)
        send_count[i] = int(cp.sum(I))
        send_buffer[offset : offset + send_count[i], :] = arr[:count, :][I, :]
        offset += send_count[i]
    # Figure out how much to receive from the others
    mpi_comm.Alltoall((send_count, 1), (recv_count, 1))
    # Perform communication of points
    mpi_comm.Alltoallv((send_buffer, 3 * send_count), (recv_buffer, 3 * recv_count))
    new_count = cp.sum(recv_count)
    arr[:new_count, :] = recv_buffer[:new_count, :]

    # Also communicate extra arrays if present
    for ea in extra_arr:
        _alltoallv_field(
            ea,
            count,
            new_count,
            send_count,
            recv_count,
            I_store,
            send_buffer,
            recv_buffer,
            mpi_comm,
            mpi_size,
        )

    return new_count


def _alltoallv_field(
    ea,
    count,
    new_count,
    send_count,
    recv_count,
    I_store,
    send_buffer,
    recv_buffer,
    mpi_comm,
    mpi_size,
):
    """
    Generalized Alltoallv for a field array ea of shape (d, N),
    using fixed-size (3, M) send/recv buffers.

    Assumes:
    - send_buffer, recv_buffer: shape (3, M), enough to hold any 3-row chunk.
    - ea is a CuPy array of shape (d, N).
    """
    d = ea.shape[1]
    assert send_buffer.shape[0] >= max(send_count), "send_buffer too small"
    assert recv_buffer.shape[0] >= max(recv_count), "recv_buffer too small"
    assert send_buffer.shape[1] == 3 and recv_buffer.shape[1] == 3

    # Process ea in 3-row chunks
    for block_start in range(0, d, 3):
        block_end = min(block_start + 3, d)
        block_size = block_end - block_start  # 1 to 3
        if block_start >= d:
            continue

        # Fill send_buffer with this block
        offset = 0
        for i in range(mpi_size):
            I = I_store[i]
            chunk = ea[:count, block_start:block_end][I, :]  # shape (Ni, block_size)
            send_buffer[offset : offset + send_count[i], :block_size] = chunk
            offset += send_count[i]

        # Communicate
        mpi_comm.Alltoallv(
            (send_buffer[:, :block_size], block_size * send_count),
            (recv_buffer[:, :block_size], block_size * recv_count),
        )

        # Write received chunk back to output
        ea[:new_count, block_start:block_end] = recv_buffer[:new_count, :block_size]


def communicate_ghost_points(
    mpi_comm,
    execution_space,
    slab,
    dist,
    L,
    arr,
    count,
    *extra_arr,
    out=None,
    buffers=None,
    base_tag=30,
):
    """
    Given a dict ``slab`` with slab information, a number
    ``dist``, periodic length ``L``, an input array ``arr``
    with ``count`` points, and optional extra arrays in
    ``extra_arr``, communicate ghost points with other MPI ranks
    and store results (owned+ghost points) in arrays given in
    ``out``, which is a tuple of size ``1+len(extra_arr)``.

    ``buffers`` may be given and will be used for sending and receiving
    MPI messages. (It is a dict object as returned by ``create_buffers()``.)

    ``mpi_comm`` is an ``MPI.Comm`` communication context.

    This function handles the communication between the ranks.

    Returns ``new_count`` which is the sum of owned points and
    ghost points.
    """
    cp = get_array_module(execution_space)

    if mpi_comm is not None:
        mpi_size = mpi_comm.Get_size()
        mpi_rank = mpi_comm.Get_rank()
    else:
        mpi_size = 1
        mpi_rank = 0

    if dist > slab["width"]:
        raise ValueError(
            f"[{mpi_rank}] Cell side length ({dist})"
            f" cannot be larger than slab width ({slab['width']})"
        )

    # Copy owned points into the output arrays
    assert len(out) == len(extra_arr) + 1
    out[0][:count, :] = arr[:count, :]
    for j, ea in enumerate(extra_arr):
        out[j + 1][:count, :] = ea[:count, :]
    new_count = count

    if mpi_size == 1:
        return new_count

    if buffers is None:
        buffers = create_buffers(arr.shape[0], mpi_comm)

    # Determine which points to send as ghost points, i.e.,
    # points that are within ``dist`` to the slab interface.
    view = arr[:count, 0]
    I_left = (view >= slab["left"]) & (view < slab["left"] + dist)
    I_right = (view >= slab["right"] - dist) & (view < slab["right"])
    i_left = (mpi_rank - 1) % mpi_size
    i_right = (mpi_rank + 1) % mpi_size
    send_buffer = buffers["send"]
    recv_buffer = buffers["recv"]
    status = MPI.Status()
    # Round 1: send to right, receive from left
    send_count = int(cp.sum(I_right))
    send_buffer[:send_count, :] = arr[:count, :][I_right, :]
    if mpi_rank == mpi_size - 1:
        send_buffer[:send_count, 0] -= L
    mpi_comm.Sendrecv(
        (send_buffer, 3 * send_count),
        dest=i_right,
        sendtag=base_tag,
        recvbuf=recv_buffer,
        source=i_left,
        recvtag=base_tag,
        status=status,
    )
    recv_count = status.Get_count(MPI.DOUBLE) // 3
    out[0][new_count : new_count + recv_count, :] = recv_buffer[:recv_count, :]

    for j, ea in enumerate(extra_arr):
        recv_check = _sendrecv_field(
            ea,
            I_right,
            count,
            send_buffer,
            recv_buffer,
            send_rank=i_right,
            recv_rank=i_left,
            tag=base_tag + j + 1,
            mpi_comm=mpi_comm,
            status=status,
            new_count=new_count,
            out=out[j + 1],
            execution_space=execution_space,
        )
        assert recv_check == recv_count

    new_count += recv_count

    # Round 2: send to left, receive from right
    send_count = int(cp.sum(I_left))
    send_buffer[:send_count, :] = arr[:count, :][I_left, :]
    if mpi_rank == 0:
        send_buffer[:send_count, 0] += L
    mpi_comm.Sendrecv(
        (send_buffer, 3 * send_count),
        dest=i_left,
        sendtag=base_tag,
        recvbuf=recv_buffer,
        source=i_right,
        recvtag=base_tag,
        status=status,
    )
    recv_count = status.Get_count(MPI.DOUBLE) // 3
    out[0][new_count : new_count + recv_count, :] = recv_buffer[:recv_count, :]

    for j, ea in enumerate(extra_arr):
        recv_check = _sendrecv_field(
            ea,
            I_left,
            count,
            send_buffer,
            recv_buffer,
            send_rank=i_left,
            recv_rank=i_right,
            tag=base_tag + j + 1,
            mpi_comm=mpi_comm,
            status=status,
            new_count=new_count,
            out=out[j + 1],
            execution_space=execution_space,
        )
        assert recv_check == recv_count

    new_count += recv_count
    return new_count


def _sendrecv_field(
    ea,
    I_mask,
    count,
    send_buffer,
    recv_buffer,
    send_rank,
    recv_rank,
    tag,
    mpi_comm,
    status,
    new_count,
    out,
    execution_space,
):
    """
    Generalized Sendrecv for ea of shape (N, d), using fixed (M, 3) buffers.

    Parameters:
    - ea: (N, d) CuPy array (row-major: each row is a vector)
    - I_mask: boolean CuPy array of shape (count,) to select rows to send
    - count: number of active points in ea
    - send_buffer, recv_buffer: (M, 3) CuPy arrays
    - send_rank, recv_rank: MPI ranks
    - tag: MPI tag
    - mpi_comm: MPI communicator
    - status: MPI.Status()
    - new_count: index offset to write received data into out
    - out: (N, d) output array to write into
    """
    cp = get_array_module(execution_space)
    d = ea.shape[1]
    send_count = int(cp.sum(I_mask))

    for block_start in range(0, d, 3):
        block_end = min(block_start + 3, d)
        block_size = block_end - block_start

        # Fill send buffer with selected rows and columns
        send_buffer[:send_count, :block_size] = ea[:count, :][
            I_mask, block_start:block_end
        ]

        mpi_comm.Sendrecv(
            (send_buffer[:, :block_size], block_size * send_count),
            dest=send_rank,
            sendtag=tag,
            recvbuf=(recv_buffer[:, :block_size], block_size * recv_buffer.shape[0]),
            source=recv_rank,
            recvtag=tag,
            status=status,
        )

        recv_actual = status.Get_count(MPI.DOUBLE) // block_size
        out[new_count : new_count + recv_actual, block_start:block_end] = recv_buffer[
            :recv_actual, :block_size
        ]

    return recv_actual


def communicate_grid_ghost_points(
    distance, arr, buffers=None, mpi_comm=None, base_tag=40
):
    """
    Given a grid ``arr``, communicate ghost grid points between the MPI
    ranks.

    ``buffers`` may be given and will be used for sending and receiving
    MPI messages. (It is a dict object as returned by ``create_buffers()``.)

    ``mpi_comm`` is an ``MPI.Comm`` communication context.

    This function handles the communication between the ranks. It
    also updates the values in ``arr``.

    Returns ``a``, which is the number of ghost grid points along
    the x direction divided by 2 (i.e., there are ``a`` ghosts to
    the left and the same amount to the right).
    """
    if mpi_comm is not None:
        mpi_size = mpi_comm.Get_size()
        mpi_rank = mpi_comm.Get_rank()
    else:
        mpi_size = 1
        mpi_rank = 0

    a = distance // 2

    if mpi_size == 1:
        arr[0:a, :, :, :] = arr[
            -2 * a : -a, :, :, :
        ]  # send to right, receive from left
        arr[-a:, :, :, :] = arr[a : 2 * a, :, :, :]  # send to left, receive from right
        return a

    # Send grid ghost points
    i_left = (mpi_rank - 1) % mpi_size
    i_right = (mpi_rank + 1) % mpi_size
    send_buffer = buffers["send"]
    recv_buffer = buffers["recv"]
    status = MPI.Status()
    count = a * np.prod(arr.shape[1:]) // 3
    # Round 1: send to right, receive from left
    if count >= send_buffer.shape[0]:
        raise BufferError(
            f"attempting to send {(int(count), 3)} items,"
            f" but send buffer can only accommodate {send_buffer.shape}."
        )
    send_buffer[:count, :] = arr[-2 * a : -a, :, :, :].reshape([-1, 3])
    mpi_comm.Sendrecv(
        (send_buffer, 3 * count),
        dest=i_right,
        sendtag=base_tag,
        recvbuf=recv_buffer,
        source=i_left,
        recvtag=base_tag,
        status=status,
    )
    assert count == status.Get_count(MPI.DOUBLE) // 3
    arr[0:a, :, :, :] = recv_buffer[:count, :].reshape([a, *arr.shape[1:]])
    # Round 2: send to left, receive from right
    send_buffer[:count, :] = arr[a : 2 * a, :, :, :].reshape([-1, 3])
    mpi_comm.Sendrecv(
        (send_buffer, 3 * count),
        dest=i_left,
        sendtag=base_tag,
        recvbuf=recv_buffer,
        source=i_right,
        recvtag=base_tag,
        status=status,
    )
    assert count == status.Get_count(MPI.DOUBLE) // 3
    arr[-a:, :, :, :] = recv_buffer[:count, :].reshape([a, *arr.shape[1:]])

    return a


def gather_points(mpi_comm, execution_space, arr, nt, buffers=None):
    """
    Gather a distributed array ``arr`` with ``nt`` points locally
    onto the first MPI rank.

    ``buffers`` may be given and will be used for sending and receiving
    MPI messages. (It is a dict object as returned by ``create_buffers()``.)

    ``mpi_comm`` is an optional ``MPI.Comm`` communication context.

    This function handles the communication between the rank.

    Returns ``new_nt``, the global number of target points (all
    gathered on the first MPI rank).
    """
    if mpi_comm is not None:
        mpi_size = mpi_comm.Get_size()
        mpi_rank = mpi_comm.Get_rank()
    else:
        return nt

    if buffers is None:
        buffers = create_buffers(mpi_comm, execution_space, arr.shape[0])

    d = arr.shape[1]
    # Gather data from others
    send_count = buffers["all_to_all_send_count"]
    recv_count = buffers["all_to_all_recv_count"]
    send_buffer = buffers["send"]
    recv_buffer = buffers["recv"]
    # Send counts first
    send_count[0] = nt
    mpi_comm.Gather((send_count, 1), (recv_count, 1))
    # Then send data
    if mpi_rank == 0:
        total_recv = int(np.sum(recv_count))
        for block_start in range(0, d, 3):
            block_end = min(block_start + 3, d)
            block_size = block_end - block_start

            # Pack local data into send buffer
            send_buffer[:nt, :block_size] = arr[:nt, block_start:block_end]

            # Compute displacements in-place
            displs = np.zeros_like(recv_count)
            displs[1:] = np.cumsum(recv_count[:-1])

            mpi_comm.Gatherv(
                (send_buffer[:, :block_size], block_size * nt),
                (recv_buffer[:, :block_size], block_size * recv_count),
                root=0,
            )

            # Copy gathered data into output array
            arr[:total_recv, block_start:block_end] = recv_buffer[
                :total_recv, :block_size
            ]

        return total_recv

    else:
        for block_start in range(0, d, 3):
            block_end = min(block_start + 3, d)
            block_size = block_end - block_start
            send_buffer[:nt, :block_size] = arr[:nt, block_start:block_end]
            mpi_comm.Gatherv(
                (send_buffer[:, :block_size], block_size * nt), None, root=0
            )

        return 0
