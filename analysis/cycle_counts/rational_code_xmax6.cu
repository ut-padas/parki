__device__ double rational_erfoverx_M3(double x)
{
    /* Horner's rule uses M-1 FMAs each for the numerator and denominator,
     * followed by a single division.
     */
    // This function uses M=3 and values for x in [0, 6].
    double numerator = 0.08039436564504378;
    double denominator = 1.0;
    numerator = 0.21008417583942413 + x*numerator;
    denominator = 0.08047821355784376 + x*denominator;
    numerator = 2.425648437319998 + x*numerator;
    denominator = 2.1566506791187425 + x*denominator;
    return numerator / denominator;
}

__device__ double rational_erfoverx_M4(double x)
{
    /* Horner's rule uses M-1 FMAs each for the numerator and denominator,
     * followed by a single division.
     */
    // This function uses M=4 and values for x in [0, 6].
    double numerator = -0.19739041253250664;
    double denominator = 1.0;
    numerator = 3.8072149172176073 + x*numerator;
    denominator = 9.745744472484256 + x*denominator;
    numerator = -4.50008139460999 + x*numerator;
    denominator = -3.6334196577770617 + x*denominator;
    numerator = 27.094951043173964 + x*numerator;
    denominator = 24.0001106059486 + x*denominator;
    return numerator / denominator;
}

__device__ double rational_erfoverx_M5(double x)
{
    /* Horner's rule uses M-1 FMAs each for the numerator and denominator,
     * followed by a single division.
     */
    // This function uses M=5 and values for x in [0, 6].
    double numerator = 0.022270856623331656;
    double denominator = 1.0;
    numerator = 0.5788396695695298 + x*numerator;
    denominator = -1.092634998999828 + x*denominator;
    numerator = 2.0160212108086104 + x*numerator;
    denominator = 6.527153642342746 + x*denominator;
    numerator = -4.698950393810701 + x*numerator;
    denominator = -4.178494188985383 + x*denominator;
    numerator = 15.734296484287919 + x*numerator;
    denominator = 13.944508943675315 + x*denominator;
    return numerator / denominator;
}

__device__ double rational_erfoverx_M6(double x)
{
    /* Horner's rule uses M-1 FMAs each for the numerator and denominator,
     * followed by a single division.
     */
    // This function uses M=6 and values for x in [0, 6].
    double numerator = 0.3027443887834594;
    double denominator = 1.0;
    numerator = -6.2351736307511 + x*numerator;
    denominator = -73.16618843075116 + x*denominator;
    numerator = -2.2110211291318738 + x*numerator;
    denominator = 147.61876383399454 + x*denominator;
    numerator = -218.37609783483302 + x*numerator;
    denominator = -523.4423584474714 + x*denominator;
    numerator = 525.7595454392725 + x*numerator;
    denominator = 465.73772281942144 + x*denominator;
    numerator = -1122.8730210680164 + x*numerator;
    denominator = -995.116452444186 + x*denominator;
    return numerator / denominator;
}

__device__ double rational_erfoverx_M7(double x)
{
    /* Horner's rule uses M-1 FMAs each for the numerator and denominator,
     * followed by a single division.
     */
    // This function uses M=7 and values for x in [0, 6].
    double numerator = 0.012498221645209405;
    double denominator = 1.0;
    numerator = 0.6398920260233218 + x*numerator;
    denominator = -4.3836563741632055 + x*denominator;
    numerator = 0.025135277217891674 + x*numerator;
    denominator = 23.726866980985275 + x*denominator;
    numerator = -6.038192474653101 + x*numerator;
    denominator = -51.36163017242252 + x*denominator;
    numerator = 68.47971247959043 + x*numerator;
    denominator = 133.7285441367732 + x*denominator;
    numerator = -154.46432805162493 + x*numerator;
    denominator = -136.8955949748263 + x*denominator;
    numerator = 247.0109892709948 + x*numerator;
    denominator = 218.9078529356488 + x*denominator;
    return numerator / denominator;
}

__device__ double rational_erfoverx_M8(double x)
{
    /* Horner's rule uses M-1 FMAs each for the numerator and denominator,
     * followed by a single division.
     */
    // This function uses M=8 and values for x in [0, 6].
    double numerator = -0.004197852162720412;
    double denominator = 1.0;
    numerator = 1.1666918046858274 + x*numerator;
    denominator = 3.8571170577029683 + x*denominator;
    numerator = 1.024385084574238 + x*numerator;
    denominator = -20.374031278828735 + x*denominator;
    numerator = 6.64159596734943 + x*numerator;
    denominator = 119.1165125768329 + x*denominator;
    numerator = -39.532674749221435 + x*numerator;
    denominator = -247.30320522794472 + x*denominator;
    numerator = 341.71940634971384 + x*numerator;
    denominator = 635.8763496433982 + x*denominator;
    numerator = -717.4784552071403 + x*numerator;
    denominator = -635.8511003874088 + x*denominator;
    numerator = 1127.2220125717724 + x*numerator;
    denominator = 998.9745203222853 + x*denominator;
    return numerator / denominator;
}

__device__ double rational_erfoverx_M10(double x)
{
    /* Horner's rule uses M-1 FMAs each for the numerator and denominator,
     * followed by a single division.
     */
    // This function uses M=10 and values for x in [0, 6].
    double numerator = 0.0008363974145758947;
    double denominator = 1.0;
    numerator = 0.969724533817759 + x*numerator;
    denominator = -4.283419676025177 + x*denominator;
    numerator = -3.829290742606836 + x*numerator;
    denominator = 16.008397786987487 + x*denominator;
    numerator = 12.580237144974301 + x*numerator;
    denominator = -4.220264294446872 + x*denominator;
    numerator = 5.97191799100089 + x*numerator;
    denominator = -47.181727410371295 + x*denominator;
    numerator = -6.597167659754986 + x*numerator;
    denominator = 455.85436336247676 + x*denominator;
    numerator = -74.25996059862418 + x*numerator;
    denominator = -994.271262621717 + x*denominator;
    numerator = 1395.2133266888643 + x*numerator;
    denominator = 2725.429384523586 + x*denominator;
    numerator = -3142.9994671356817 + x*numerator;
    denominator = -2785.410769660202 + x*denominator;
    numerator = 5040.314461058636 + x*numerator;
    denominator = 4466.86238860197 + x*denominator;
    return numerator / denominator;
}

__device__ double rational_erfoverx_M12(double x)
{
    /* Horner's rule uses M-1 FMAs each for the numerator and denominator,
     * followed by a single division.
     */
    // This function uses M=12 and values for x in [0, 6].
    double numerator = -0.0007600747930774759;
    double denominator = 1.0;
    numerator = 1.036038035840463 + x*numerator;
    denominator = -5.332430916251075 + x*denominator;
    numerator = -6.104998890745315 + x*numerator;
    denominator = 19.411210663701258 + x*denominator;
    numerator = 29.323929518048296 + x*numerator;
    denominator = 9.66313848020263 + x*denominator;
    numerator = -75.43298524891813 + x*numerator;
    denominator = -224.0748194074655 + x*denominator;
    numerator = 295.2941026680057 + x*numerator;
    denominator = 1429.3265501291187 + x*denominator;
    numerator = -921.5287538137183 + x*numerator;
    denominator = -4284.569221285354 + x*denominator;
    numerator = 3896.0125416570604 + x*numerator;
    denominator = 12132.227304717706 + x*denominator;
    numerator = -10377.682146595493 + x*numerator;
    denominator = -21287.802335374236 + x*denominator;
    numerator = 27741.53793045295 + x*numerator;
    denominator = 39121.4082219599 + x*denominator;
    numerator = -40929.15742668267 + x*numerator;
    denominator = -36272.52131546322 + x*denominator;
    numerator = 49206.736515780736 + x*numerator;
    denominator = 43608.33481380129 + x*denominator;
    return numerator / denominator;
}

__device__ double rational_erfoverx_M13(double x)
{
    /* Horner's rule uses M-1 FMAs each for the numerator and denominator,
     * followed by a single division.
     */
    // This function uses M=13 and values for x in [0, 6].
    double numerator = 0.0006857688428852644;
    double denominator = 1.0;
    numerator = 0.9653113634505226 + x*numerator;
    denominator = -6.316290662667749 + x*denominator;
    numerator = -5.52964503635217 + x*numerator;
    denominator = 35.0463095969395 + x*denominator;
    numerator = 24.540535366752035 + x*numerator;
    denominator = -92.39209654389332 + x*denominator;
    numerator = -1.235015033975131 + x*numerator;
    denominator = 318.3087976275932 + x*denominator;
    numerator = -215.75607026942978 + x*numerator;
    denominator = -557.4646398258508 + x*denominator;
    numerator = 1571.8821079031252 + x*numerator;
    denominator = 2600.553147658497 + x*denominator;
    numerator = -3186.0326594076496 + x*numerator;
    denominator = -5312.765433768676 + x*denominator;
    numerator = 6510.719494015144 + x*numerator;
    denominator = 23804.65092137442 + x*denominator;
    numerator = -4395.967913372465 + x*numerator;
    denominator = -39598.99436393835 + x*denominator;
    numerator = 53036.06768897573 + x*numerator;
    denominator = 118019.74938174247 + x*denominator;
    numerator = -120860.10608022155 + x*numerator;
    denominator = -107109.48023277304 + x*denominator;
    numerator = 240404.8747280965 + x*numerator;
    denominator = 213053.2729941808 + x*denominator;
    return numerator / denominator;
}

__device__ double rational_erfoverx_M14(double x)
{
    /* Horner's rule uses M-1 FMAs each for the numerator and denominator,
     * followed by a single division.
     */
    // This function uses M=14 and values for x in [0, 6].
    double numerator = -0.0031316629038530253;
    double denominator = 1.0;
    numerator = 1.178470790614383 + x*numerator;
    denominator = 0.4771292428369437 + x*denominator;
    numerator = -4.160897941243252 + x*numerator;
    denominator = -35.24248137647824 + x*denominator;
    numerator = 37.358969952366635 + x*numerator;
    denominator = 509.8975574650368 + x*denominator;
    numerator = -251.72158392868985 + x*numerator;
    denominator = -2921.0467973055092 + x*denominator;
    numerator = 2717.7052519543918 + x*numerator;
    denominator = 15070.657016792851 + x*denominator;
    numerator = -15210.77074099742 + x*numerator;
    denominator = -53447.371924635816 + x*denominator;
    numerator = 66717.84082723015 + x*numerator;
    denominator = 183172.19896398715 + x*denominator;
    numerator = -179477.11194492914 + x*numerator;
    denominator = -458407.6960956476 + x*denominator;
    numerator = 435774.96204571426 + x*numerator;
    denominator = 1.1623709109791024e6 + x*denominator;
    numerator = -873429.3991061687 + x*numerator;
    denominator = -2.0140043668290335e6 + x*denominator;
    numerator = 2.4417914281184585e6 + x*numerator;
    denominator = 3.8094315052043116e6 + x*denominator;
    numerator = -4.197393525175254e6 + x*numerator;
    denominator = -3.7198431587273185e6 + x*denominator;
    numerator = 5.570075163167838e6 + x*numerator;
    denominator = 4.936350586395006e6 + x*denominator;
    return numerator / denominator;
}

