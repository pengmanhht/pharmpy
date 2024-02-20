import re
import shutil
from io import StringIO

import numpy as np
import pytest
from pytest import approx

import pharmpy.tools as tools
from pharmpy.deps import pandas as pd
from pharmpy.tools import read_modelfit_results
from pharmpy.tools.frem.models import calculate_parcov_inits, create_model3b
from pharmpy.tools.frem.results import (
    calculate_results,
    calculate_results_using_bipp,
    get_params,
    psn_frem_results,
)
from pharmpy.tools.frem.tool import check_covariates
from pharmpy.tools.psn_helpers import create_results


def test_check_covariates(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'pheno_real.mod')
    newcov = check_covariates(model, ['WGT', 'APGR'])
    assert newcov == ['WGT', 'APGR']
    newcov = check_covariates(model, ['APGR', 'WGT'])
    assert newcov == ['APGR', 'WGT']
    data = model.dataset
    data['NEW'] = data['WGT']
    model = model.replace(dataset=data)
    with pytest.warns(UserWarning):
        newcov = check_covariates(model, ['APGR', 'WGT', 'NEW'])
    assert newcov == ['APGR', 'WGT']
    with pytest.warns(UserWarning):
        newcov = check_covariates(model, ['NEW', 'APGR', 'WGT'])
    assert newcov == ['NEW', 'APGR']


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_check_covariates_mult_warns(load_model_for_test, testdata):
    # These are separated because capturing the warnings did not work.
    # Possibly because more than one warning is issued
    model = load_model_for_test(testdata / 'nonmem' / 'pheno_real.mod')
    newcov = check_covariates(model, ['FA1', 'FA2'])
    assert newcov == []


def test_parcov_inits(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'frem' / 'pheno' / 'model_3.mod')
    res = read_modelfit_results(testdata / 'nonmem' / 'frem' / 'pheno' / 'model_3.mod')
    params = calculate_parcov_inits(model, res.individual_estimates, 2)
    assert params == approx(
        {
            'OMEGA_3_1': 0.02560327,
            'OMEGA_3_2': -0.001618381,
            'OMEGA_4_1': -0.06764814,
            'OMEGA_4_2': 0.02350935,
        }
    )


def test_create_model3b(load_model_for_test, testdata):
    model3 = load_model_for_test(testdata / 'nonmem' / 'frem' / 'pheno' / 'model_3.mod')
    model3_res = read_modelfit_results(testdata / 'nonmem' / 'frem' / 'pheno' / 'model_3.mod')
    model1b = load_model_for_test(testdata / 'nonmem' / 'pheno_real.mod')
    model3b = create_model3b(model1b, model3, model3_res, 2)
    pset = model3b.parameters
    assert pset['OMEGA_3_1'].init == approx(0.02560327)
    assert pset['POP_CL'].init == 0.00469555
    assert model3b.name == 'model_3b'


def test_bipp_covariance(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'frem' / 'pheno' / 'model_4.mod')
    res = read_modelfit_results(testdata / 'nonmem' / 'frem' / 'pheno' / 'model_4.mod')
    res = calculate_results_using_bipp(
        model, res, continuous=['APGR', 'WGT'], categorical=[], seed=9532
    )
    assert res


def test_frem_results_pheno(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'frem' / 'pheno' / 'model_4.mod')
    res = read_modelfit_results(testdata / 'nonmem' / 'frem' / 'pheno' / 'model_4.mod')
    rng = np.random.default_rng(39)
    res = calculate_results(
        model, res, continuous=['APGR', 'WGT'], categorical=[], samples=10, seed=rng
    )

    correct = """parameter,covariate,condition,p5,mean,p95
CL,APGR,5th,0.972200,1.099294,1.242870
CL,APGR,95th,0.901919,0.958505,1.013679
CL,WGT,5th,0.864055,0.941833,1.003508
CL,WGT,95th,0.993823,1.138392,1.346290
V,APGR,5th,0.818275,0.900856,0.980333
V,APGR,95th,1.009562,1.052237,1.099986
V,WGT,5th,0.957804,0.994109,1.030915
V,WGT,95th,0.940506,1.014065,1.091648
"""

    correct = pd.read_csv(StringIO(correct), index_col=[0, 1, 2])
    correct.index.set_names(['parameter', 'covariate', 'condition'], inplace=True)
    pd.testing.assert_frame_equal(res.covariate_effects, correct)

    correct = """ID,parameter,observed,p5,p95
1,CL,0.9747876089262426,0.9623876223423867,0.9824852674876945
1,V,1.0220100411225312,1.0023922181039566,1.0210291755653986
2,CL,0.9322191565835464,0.879772943050749,0.9870177704564436
2,V,1.0863063389527468,1.0106054760663574,1.0949107930978257
3,CL,1.0091536521780653,0.9983634898455471,1.018968935499065
3,V,0.9872706260693193,0.9844268334767777,0.9985306191354839
4,CL,0.9606207744865287,0.8966200610185175,0.9788305801683045
4,V,1.0035010263500566,0.9663822687437207,1.0253995817536679
5,CL,0.9747876089262426,0.9623876223423867,0.9824852674876945
5,V,1.0220100411225312,1.0023922181039566,1.0210291755653986
6,CL,1.0109608882054792,0.95675167702286,1.047032014139463
6,V,0.9641361555839656,0.9426547826305651,0.9971274446069774
7,CL,0.9944873258008428,0.9234085517155061,1.0329306928525204
7,V,0.9693908196899773,0.9377215638220676,1.005787746465714
8,CL,0.958903488487699,0.9297581756071042,0.9698533840376546
8,V,1.0275801252314685,0.9995096153857336,1.0305284160928718
9,CL,0.9493585916770979,0.9168219777534304,0.9781229378966473
9,V,1.0551004552760428,1.0074271830514618,1.0558192686142556
10,CL,0.9747876089262426,0.9623876223423867,0.9824852674876945
10,V,1.0220100411225312,1.0023922181039566,1.0210291755653986
11,CL,0.958903488487699,0.9297581756071042,0.9698533840376546
11,V,1.0275801252314685,0.9995096153857336,1.0305284160928718
12,CL,0.9927094983849412,0.964423399504378,1.0049743061538365
12,V,0.9926513892731682,0.9792468738482092,1.0050990195407465
13,CL,0.976533341560172,0.9298917166331006,0.9915919601050044
13,V,0.9980614650127685,0.9734701845573707,1.014970713377564
14,CL,0.951058782653966,0.9134209062993411,0.9640823778650414
14,V,1.0303765312253497,0.9980598360399148,1.0355596311429953
15,CL,0.9668129564881183,0.946396890356916,0.9756598929662376
15,V,1.0247912889635773,1.0009493481389617,1.0255263136231711
16,CL,0.9095265185860849,0.8596675559459472,0.9496967076081981
16,V,1.0951991931031655,1.0132776214955752,1.0943338294249254
17,CL,1.0026902472906303,0.9399291242693675,1.0399572629061364
17,V,0.9667599267916591,0.9401828423267352,1.0008887085836748
18,CL,0.9944873258008428,0.9234085517155061,1.0329306928525204
18,V,0.9693908196899773,0.9377215638220676,1.005787746465714
19,CL,1.1053968956987084,0.9550940894449401,1.252630489586505
19,V,0.8533833334063132,0.8137505765058551,0.9831214378016516
20,CL,0.9845882285317421,0.946997120743552,0.9982223740073822
20,V,0.9953527419235736,0.9766736357263162,1.0100205095501023
21,CL,1.0073495639382477,0.9911227412088595,1.044387148904489
21,V,1.010960288562291,0.9946666839380782,1.0280977419525212
22,CL,0.95718931798836,0.9239134454923038,0.990767474177394
22,V,1.0522369332956463,1.0065490452129744,1.056769129368429
23,CL,1.2458833674993706,1.174095979524402,1.4471022995695604
23,V,0.8590844344554049,0.8511772373661953,0.9978047397479437
24,CL,1.2898066624107463,1.2042629850784445,1.5119655998604768
24,V,0.8298831532047879,0.8263339770363893,0.9920863584513262
25,CL,1.078488455291716,0.9098851631064538,1.225407606892303
25,V,0.8603694590532166,0.8074080480904899,0.9856154957740352
26,CL,1.098674754163057,1.0106076147912186,1.391954855775848
26,V,1.0288378820849053,0.9418188471476597,1.1554219798982115
27,CL,1.070797186680841,1.0497848923764674,1.1095595928316735
27,V,0.9459681173444604,0.9476795247485684,0.9950741922847095
28,CL,1.0719298253304665,0.9904819562406533,1.3176274576021954
28,V,1.03726038617135,0.9555429217552073,1.1460077037451706
29,CL,0.9432781990804444,0.8973792501310945,0.9583465711235544
29,V,1.033180567003963,0.9960458633951124,1.040714440250996
30,CL,0.9810711118911185,0.9449009310699679,1.0344015681825607
30,V,1.0436929362812277,1.0039329214277344,1.0652073284592802
31,CL,0.9493585916770979,0.9168219777534304,0.9781229378966473
31,V,1.0551004552760428,1.0074271830514618,1.0558192686142556
32,CL,1.107736990310448,1.017366137348099,1.4176689131640372
32,V,1.0260456715775965,0.9372967141871299,1.158587315986331
33,CL,0.9730450546630892,0.9382619333344786,1.0165736891038182
33,V,1.0465331908863182,1.0048019204607868,1.0623824400913477
34,CL,1.0815633528084572,1.0321958697700877,1.1412768058833191
34,V,0.9212939726059592,0.9200323034240248,0.988963216508514
35,CL,1.1248963316842946,1.0896346701791253,1.2343223846182518
35,V,0.930668271710216,0.9207989580569778,1.003672734199572
36,CL,1.0361843083002586,1.00221039689929,1.0701140338462605
36,V,0.9563075137562995,0.9501349320058959,0.9945680777536055
37,CL,0.9095265185860849,0.8596675559459472,0.9496967076081981
37,V,1.0951991931031655,1.0132776214955752,1.0943338294249254
38,CL,0.941591971811965,0.9097851719695764,0.9656476681586792
38,V,1.057971753742054,1.008294200694932,1.0570460812274063
39,CL,0.9382282786328352,0.863978786528425,1.035323710460379
39,V,1.1093533335978631,1.0111358019986683,1.1465425834388718
40,CL,1.0571250130931833,0.9581487136615587,1.1444659341348051
40,V,0.9070708069040182,0.8758299821223936,0.9900800887894693
41,CL,0.999108524985206,0.9844282618752698,1.0258297357380268
41,V,1.0137114661503888,0.9995253738170382,1.0253723417738896
42,CL,1.0372807133026336,0.9639930739237731,1.2247756092969586
42,V,1.048597477200077,0.9742146897963039,1.133644558392112
43,CL,1.0963536766193422,0.942906018662724,1.2430275278873713
43,V,0.8557057052065501,0.8116272866638132,0.9839498555735701
44,CL,0.9747876089262426,0.9623876223423867,0.9824852674876945
44,V,1.0220100411225312,1.0023922181039566,1.0210291755653986
45,CL,1.059017894749882,0.9172185488462667,1.1761347712395736
45,V,0.8858157120486193,0.838722698055201,0.9886853338113423
46,CL,0.9262487191851418,0.8897940034550293,0.9468609378746285
46,V,1.0637378541056135,1.0053811001126487,1.0618091096304203
47,CL,1.0203778571306483,0.9510168403689275,1.1808779605388167
47,V,1.0543126075153102,0.9837140275640088,1.1275426462036497
48,CL,0.8963081434656491,0.8289037114264994,0.923248384252418
48,V,1.0753645249641806,0.9995929717263163,1.0823351064416151
49,CL,0.941591971811965,0.9097851719695764,0.9656476681586792
49,V,1.057971753742054,1.008294200694932,1.0570460812274063
50,CL,0.976533341560172,0.9298917166331006,0.9915919601050044
50,V,0.9980614650127685,0.9734701845573707,1.014970713377564
51,CL,0.887386262989466,0.8348776008389295,0.9195007177004489
51,V,1.104164853369307,1.0098626419201524,1.0995841187514204
52,CL,0.9355612682293925,0.8816278276092833,0.9526457840151445
52,V,1.0359922336014173,0.9925407119469869,1.0463332767905136
53,CL,0.9730450546630892,0.9382619333344786,1.0165736891038182
53,V,1.0465331908863182,1.0048019204607868,1.0623824400913477
54,CL,0.9810711118911185,0.9449009310699679,1.0344015681825607
54,V,1.0436929362812277,1.0039329214277344,1.0652073284592802
55,CL,1.0295480825430916,0.9489431971840638,1.0908624641438185
55,V,0.9364397440565791,0.9074358325804165,0.9940178908810062
56,CL,0.9881180733519868,0.868368787557875,1.0544960994592296
56,V,0.9492512613562971,0.8956645356453113,1.0114121853036042
57,CL,1.060138503325679,1.043458933657303,1.1198727965741848
57,V,0.9713027400777485,0.9613644126949645,1.009755277835067
58,CL,0.9493585916770979,0.9168219777534304,0.9781229378966473
58,V,1.0551004552760428,1.0074271830514618,1.0558192686142556
59,CL,0.976533341560172,0.9298917166331006,0.9915919601050044
59,V,0.9980614650127685,0.9734701845573707,1.014970713377564
"""
    correct = pd.read_csv(StringIO(correct), dtype={0: 'int32'}, index_col=[0, 1])
    correct.index.set_names(['ID', 'parameter'], inplace=True)
    pd.testing.assert_frame_equal(res.individual_effects, correct)

    correct = """parameter,covariate,sd_observed,sd_5th,sd_95th
CL,none,0.19836380718266122,0.10698386364464521,0.22813605494479994
CL,APGR,0.1932828383897819,0.08207800471169897,0.22738951605057137
CL,WGT,0.19363776172900196,0.10259365732821585,0.19906614312476859
CL,all,0.1851006246151042,0.06925915743342524,0.1897192131955216
V,none,0.16105092362355455,0.12600993671999713,0.18079489759700668
V,APGR,0.1468832868065463,0.11406607129463658,0.1704182899316319
V,WGT,0.16104200315990183,0.12203994522797203,0.18040105423522765
V,all,0.14572521381314374,0.11146577839548052,0.16976758171177983
"""
    correct = pd.read_csv(StringIO(correct), index_col=[0, 1])
    correct.index.set_names(['parameter', 'covariate'], inplace=True)
    pd.testing.assert_frame_equal(res.unexplained_variability, correct)

    correct = pd.DataFrame(
        {
            'p5': [1.0, 0.7],
            'mean': [6.423729, 1.525424],
            'p95': [9.0, 3.2],
            'stdev': [2.237636, 0.704565],
            'ref': [6.423729, 1.525424],
            'categorical': [False, False],
            'other': [np.nan, np.nan],
        },
        index=['APGR', 'WGT'],
    )
    correct.index.name = 'covariate'
    pd.testing.assert_frame_equal(res.covariate_statistics, correct)


def test_frem_results_pheno_categorical(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'frem' / 'pheno_cat' / 'model_4.mod')
    res = read_modelfit_results(testdata / 'nonmem' / 'frem' / 'pheno_cat' / 'model_4.mod')
    rng = np.random.default_rng(8978)
    res = calculate_results(
        model, res, continuous=['WGT'], categorical=['APGRX'], samples=10, seed=rng
    )

    correct = """parameter,covariate,condition,p5,mean,p95
CL,WGT,5th,0.8888806479928386,0.9344068474824363,1.0429216842428766
CL,WGT,95th,0.9275380047890988,1.1598301793623595,1.2703988351195366
CL,APGRX,other,0.9915508260440394,1.0837646491987887,1.2173634348518769
V,WGT,5th,0.9765936229688581,1.0120003772784190,1.0739756346907814
V,WGT,95th,0.8694233922131289,0.9798663683220742,1.0494977661151397
V,APGRX,other,0.8545033358481884,0.9090756330508469,0.9559616221673316
"""

    correct = pd.read_csv(StringIO(correct), index_col=[0, 1, 2])
    correct.index.set_names(['parameter', 'covariate', 'condition'], inplace=True)
    pd.testing.assert_frame_equal(res.covariate_effects, correct)

    correct = """ID,parameter,observed,p5,p95
1,CL,0.9912884661387944,0.9823252452775585,1.005189333353676
1,V,1.0011657907076288,0.9983757554340882,1.0119068964710976
2,CL,0.9982280031437448,0.9963915937346052,1.001044243672966
2,V,1.0002361858019588,0.9996705321019457,1.0023990815855455
3,CL,0.9982280031437448,0.9963915937346052,1.001044243672966
3,V,1.0002361858019588,0.9996705321019457,1.0023990815855455
4,CL,0.9573077718907862,0.9149358617636167,1.0268362244163478
4,V,1.0058267874330078,0.9919272607197847,1.0612008300863554
5,CL,0.9912884661387944,0.9823252452775585,1.005189333353676
5,V,1.0011657907076288,0.9983757554340882,1.0119068964710976
6,CL,1.1198435491217265,0.9547645645962481,1.1881158336431126
6,V,0.9085976422437901,0.855064355414568,0.957789916083065
7,CL,1.1043277095093798,0.9322066409829316,1.1887146448927077
7,V,0.9102872994311817,0.8535396960971104,0.9630156598488638
8,CL,0.9775538039511688,0.9547895069246628,1.0136621156761403
8,V,1.0030275905796406,0.99579128927331,1.0312688104638512
9,CL,0.9912884661387944,0.9823252452775585,1.005189333353676
9,V,1.0011657907076288,0.9983757554340882,1.0119068964710976
10,CL,0.9912884661387944,0.9823252452775585,1.005189333353676
10,V,1.0011657907076288,0.9983757554340882,1.0119068964710976
11,CL,0.9775538039511688,0.9547895069246628,1.0136621156761403
11,V,1.0030275905796406,0.99579128927331,1.0312688104638512
12,CL,0.984397205662633,0.9684588915495942,1.0093950894728805
12,V,1.0020962549833665,0.9970826803995897,1.021529589408315
13,CL,0.9707579766809108,0.9413143707818629,1.017990941398883
13,V,1.0039597917474488,0.9945015889322739,1.0411260478578415
14,CL,0.9707580846524858,0.9413144737110017,1.0179909617871041
14,V,1.0039597398327802,0.9945015516374013,1.0411259328046443
15,CL,0.984397205662633,0.9684588915495942,1.0093950894728805
15,V,1.0020962549833665,0.9970826803995897,1.021529589408315
16,CL,0.9775538039511688,0.9547895069246628,1.0136621156761403
16,V,1.0030275905796406,0.99579128927331,1.0312688104638512
17,CL,1.112058541995997,0.9434176273196895,1.187813357025184
17,V,0.9094420814108903,0.8543016602749455,0.9602788763435623
18,CL,1.1043277095093798,0.9322066409829316,1.1887146448927077
18,V,0.9102872994311817,0.8535396960971104,0.9630156598488638
19,CL,1.1043277095093798,0.9322066409829316,1.1887146448927077
19,V,0.9102872994311817,0.8535396960971104,0.9630156598488638
20,CL,0.9775538039511688,0.9547895069246628,1.0136621156761403
20,V,1.0030275905796406,0.99579128927331,1.0312688104638512
21,CL,1.0193394191110992,0.9889676754096377,1.0398193488163914
21,V,0.9974525520566727,0.974550209921862,1.0035650488048375
22,CL,0.9982280031437448,0.9963915937346052,1.001044243672966
22,V,1.0002361858019588,0.9996705321019457,1.0023990815855455
23,CL,1.278561145560325,1.048547708317357,1.4402183968948892
23,V,0.8927014996056475,0.7567676250067886,0.9543384498249149
24,CL,1.2875116337804728,1.0447998902777895,1.4561976503405143
24,V,0.8918726161520479,0.7501095121226353,0.9550317484220464
25,CL,1.0814559146197718,0.8993722377570724,1.1921910165209448
25,V,0.9128276994620533,0.8512581230269294,0.9863864985747671
26,CL,1.1476875524497763,0.9300837994094086,1.3244994512246249
26,V,0.9818242876491968,0.8341896563625997,1.0259255530597293
27,CL,1.175885957445256,1.0381392471762432,1.2616113746617195
27,V,0.9027084993825373,0.8431354134536979,0.9502634232171141
28,CL,1.1239176188679145,0.9393538830305941,1.2690915372667446
28,V,0.9845643436557832,0.8569600526613287,1.021943254906035
29,CL,0.9640094405028626,0.9280307976019484,1.0223821001641527
29,V,1.0048928527141423,0.9932135862088474,1.0511028079354217
30,CL,1.0193394191110992,0.9889676754096377,1.0398193488163914
30,V,0.9974525520566727,0.974550209921862,1.0035650488048375
31,CL,0.9912884661387944,0.9823252452775585,1.005189333353676
31,V,1.0011657907076288,0.9983757554340882,1.0119068964710976
32,CL,1.1557218681759203,0.9270950567611691,1.343504477895859
32,V,0.9809126526774464,0.8267754597773977,1.0272564376102078
33,CL,1.0122531534168406,0.9929339453254553,1.0251357110615875
33,V,0.9983795653196571,0.9837222019678337,1.0022651792878396
34,CL,1.1595936298853569,1.0135959499318568,1.2340764764985548
34,V,0.9043872049318215,0.8519565402856528,0.9524039777299782
35,CL,1.2261492360260002,1.0722995753248208,1.3479587202710146
35,V,0.8976910613647905,0.7984329845328006,0.9502010335597292
36,CL,1.1435270215868598,0.9896371945721567,1.2071427196257194
36,V,0.9060690340346341,0.8546398029732785,0.9545524160474242
37,CL,0.9775538039511688,0.9547895069246628,1.0136621156761403
37,V,1.0030275905796406,0.99579128927331,1.0312688104638512
38,CL,0.984397205662633,0.9684588915495942,1.0093950894728805
38,V,1.0020962549833665,0.9970826803995897,1.021529589408315
39,CL,1.026475342573432,0.9850599091939051,1.0547148816513052
39,V,0.9965263930197235,0.9654871471667734,1.004866631741699
40,CL,1.112058541995997,0.9434176273196895,1.187813357025184
40,V,0.9094420814108903,0.8543016602749455,0.9602788763435623
41,CL,1.0122531534168406,0.9929339453254553,1.0251357110615875
41,V,0.9983795653196571,0.9837222019678337,1.0022651792878396
42,CL,1.0929891262425546,0.9524390801842624,1.198827899604719
42,V,0.9882295642803754,0.8885941936936287,1.0166579437026515
43,CL,1.0966505664884518,0.9211298087918399,1.18974467557879
43,V,0.9111333089443203,0.8527784514139055,0.96981477599534
44,CL,0.9912884661387944,0.9823252452775585,1.005189333353676
44,V,1.0011657907076288,0.9983757554340882,1.0119068964710976
45,CL,1.0890265798617313,0.9101852847076254,1.1909034865933101
45,V,0.911980128603927,0.8520179094439722,0.9779878260361825
46,CL,0.9707579766809108,0.9413143707818629,1.017990941398883
46,V,1.0039597917474488,0.9945015889322739,1.0411260478578415
47,CL,1.0778449935823065,0.9593011575628277,1.165177169111004
47,V,0.990067352893127,0.9049799513007414,1.0140255662370024
48,CL,0.9440437853762491,0.889301249526966,1.0359356543164355
48,V,1.0076972748297237,0.9893596417011579,1.0817671034842347
49,CL,0.984397205662633,0.9684588915495942,1.0093950894728805
49,V,1.0020962549833665,0.9970826803995897,1.021529589408315
50,CL,0.9707579766809108,0.9413143707818629,1.017990941398883
50,V,1.0039597917474488,0.9945015889322739,1.0411260478578415
51,CL,0.9573077718907862,0.9149358617636167,1.0268362244163478
51,V,1.0058267874330078,0.9919272607197847,1.0612008300863554
52,CL,0.9573077718907862,0.9149358617636167,1.0268362244163478
52,V,1.0058267874330078,0.9919272607197847,1.0612008300863554
53,CL,1.0122531534168406,0.9929339453254553,1.0251357110615875
53,V,0.9983795653196571,0.9837222019678337,1.0022651792878396
54,CL,1.0193394191110992,0.9889676754096377,1.0398193488163914
54,V,0.9974525520566727,0.974550209921862,1.0035650488048375
55,CL,1.112058541995997,0.9434176273196895,1.187813357025184
55,V,0.9094420814108903,0.8543016602749455,0.9602788763435623
56,CL,1.0739378789217238,0.8886886227193302,1.1936073485795418
56,V,0.9136760580307196,0.8504990592696858,0.994892338927509
57,CL,1.0408973539422075,0.9774179040022573,1.0851535618204715
57,V,0.9946766605413693,0.9476823542166756,1.0074749102503402
58,CL,0.9912884661387944,0.9823252452775585,1.005189333353676
58,V,1.0011657907076288,0.9983757554340882,1.0119068964710976
59,CL,0.9707579766809108,0.9413143707818629,1.017990941398883
59,V,1.0039597917474488,0.9945015889322739,1.0411260478578415
"""

    correct = pd.read_csv(StringIO(correct), dtype={0: 'int32'}, index_col=[0, 1])
    correct.index.set_names(['ID', 'parameter'], inplace=True)
    pd.testing.assert_frame_equal(res.individual_effects, correct)

    correct = """parameter,covariate,sd_observed,sd_5th,sd_95th
CL,none,0.1876414133393799,0.1211647441717813,0.2252670442425513
CL,WGT,0.1824855585272548,0.1000742727854050,0.1989308616423449
CL,APGRX,0.1785985176170080,0.1194859118429075,0.2170526201633760
CL,all,0.1718672014845674,0.0990338088120116,0.1924954360156897
V,none,0.1509307788358624,0.1479980261201491,0.2017978735550682
V,WGT,0.1509045294791560,0.1368087748774919,0.2012032522786724
V,APGRX,0.1442982672200497,0.1410511910779913,0.1937071465967700
V,all,0.1441532460182698,0.1281115989110297,0.1928352776311023
"""

    correct = pd.read_csv(StringIO(correct), index_col=[0, 1])
    correct.index.set_names(['parameter', 'covariate'], inplace=True)
    pd.testing.assert_frame_equal(res.unexplained_variability, correct)

    correct = pd.DataFrame(
        {
            'p5': [0.7, 0],
            'mean': [1.525424, 0.711864],
            'p95': [3.2, 1],
            'stdev': [0.704565, 0.456782],
            'ref': [1.525424, 1.0],
            'categorical': [False, True],
            'other': [np.nan, 0],
        },
        index=['WGT', 'APGRX'],
    )
    correct.index.name = 'covariate'
    pd.testing.assert_frame_equal(res.covariate_statistics, correct)


def test_get_params(load_model_for_test, create_model_for_test, testdata):
    model_frem = load_model_for_test(testdata / 'nonmem' / 'frem' / 'pheno' / 'model_4.mod')
    dist = model_frem.random_variables.etas[-1]
    rvs = list(dist.names)
    npars = 2

    param_names = get_params(model_frem, rvs, npars)
    assert param_names == ['CL', 'V']

    model_multiple_etas = re.sub(
        r'(V=TVV\*EXP\(ETA\(2\)\))', r'\1*EXP(ETA(3))', model_frem.model_code
    )

    model = create_model_for_test(model_multiple_etas)
    model = model.replace(dataset=model_frem.dataset)
    dist = model.random_variables.etas[-1]
    rvs = list(dist.names)
    npars = 3

    param_names = get_params(model, rvs, npars)
    assert param_names == ['CL', 'V(1)', 'V(2)']

    model_separate_declare = re.sub(
        r'(V=TVV\*EXP\(ETA\(2\)\))',
        'ETA2=ETA(2)\n      V=TVV*EXP(ETA2)',
        model_frem.model_code,
    )

    model = create_model_for_test(model_separate_declare)
    model = model.replace(dataset=model_frem.dataset)
    dist = model.random_variables.etas[-1]
    rvs = list(dist.names)
    npars = 2

    param_names = get_params(model, rvs, npars)
    print(param_names)
    assert param_names == ['CL', 'V']


def test_psn_frem_results(testdata):
    with pytest.warns(UserWarning):
        res = psn_frem_results(testdata / 'psn' / 'frem_dir1', method='bipp')
    ofv = res.ofv['ofv']
    assert len(ofv) == 5
    assert ofv['model_1'] == approx(730.894727)
    assert ofv['model_2'] == approx(896.974324)
    assert ofv['model_3'] == approx(868.657803)
    assert ofv['model_3b'] == approx(852.803483)
    assert ofv['model_4'] == approx(753.302743)

    correct = """model type		TVCL  TVV  IVCL  OMEGA_2_1  IVV  OMEGA_3_1  OMEGA_3_2  BSV_APGR  OMEGA_4_1  OMEGA_4_2  OMEGA_4_3  BSV_WGT  SIGMA_1_1
model_1  init      0.004693   1.00916    0.030963         NaN    0.031128         NaN         NaN         NaN         NaN         NaN         NaN         NaN    0.013241
model_1  estimate  0.005818   1.44555    0.111053         NaN    0.201526         NaN         NaN         NaN         NaN         NaN         NaN         NaN    0.016418
model_2  init           NaN       NaN         NaN         NaN         NaN         NaN         NaN    1.000000         NaN         NaN    0.244579    1.000000         NaN
model_2  estimate       NaN       NaN         NaN         NaN         NaN         NaN         NaN    1.000000         NaN         NaN    0.244579    1.000000         NaN
model_3  init           NaN       NaN    0.115195    0.007066    0.209016   -0.010583    0.107027    1.000008    0.171529    0.404278    0.244448    1.002173         NaN
model_3  estimate       NaN       NaN    0.115195    0.007066    0.209016   -0.010583    0.107027    1.000010    0.171529    0.404278    0.244448    1.002170         NaN
model_3b init      0.005818   1.44555    0.125999    0.020191    0.224959   -0.012042    0.115427    1.000032    0.208475    0.415588    0.244080    1.007763    0.016418
model_3b estimate  0.005818   1.44555    0.126000    0.020191    0.224959   -0.012042    0.115427    1.000030    0.208475    0.415588    0.244080    1.007760    0.016418
model_4  init      0.005818   1.44555    0.126000    0.020191    0.224959   -0.012042    0.115427    1.000030    0.208475    0.415588    0.244080    1.007760    0.016418
model_4  estimate  0.007084   1.38635    0.220463    0.195326    0.176796    0.062712    0.117271    1.039930    0.446939    0.402075    0.249237    1.034610    0.015250
"""  # noqa E501
    correct = pd.read_csv(StringIO(correct), index_col=[0, 1], sep=r'\s+')
    pd.testing.assert_frame_equal(res.parameter_inits_and_estimates, correct, rtol=1e-4)

    pc = res.base_parameter_change
    assert len(pc) == 5
    assert pc['TVCL'] == 21.77321763763502
    assert pc['TVV'] == -4.095327038151563
    assert pc['IVCL'] == pytest.approx(98.52052623522104, abs=1e-12)
    assert pc['IVV'] == -12.271369451088198
    assert pc['SIGMA_1_1'] == pytest.approx(-7.110618417927009, abs=1e-12)

    correct = """,mean,stdev
APGR,6.42372,2.237640
WGT,1.525424,0.704565
"""
    correct = pd.read_csv(StringIO(correct), index_col=[0])
    pd.testing.assert_frame_equal(res.estimated_covariates, correct, rtol=1e-5)

    correct = """condition,parameter,CL,V
all,CL,0.025328,0.022571
all,V,0.022571,0.020115
APGR,CL,0.216681,0.188254
APGR,V,0.188254,0.163572
WGT,CL,0.027391,0.021634
WGT,V,0.021634,0.020540
"""
    correct = pd.read_csv(StringIO(correct), index_col=[0, 1])
    pd.testing.assert_frame_equal(res.parameter_variability, correct, rtol=1e-4)

    correct = """condition,parameter,APGR,WGT
all,CL,-0.020503,0.628814
all,V,0.00930905,0.544459
each,CL,0.0269498,0.613127
each,V,0.0503961,0.551581
"""

    correct = pd.read_csv(StringIO(correct), index_col=[0, 1])
    pd.testing.assert_frame_equal(res.coefficients, correct, rtol=1e-5)


def test_create_results(testdata):
    with pytest.warns(UserWarning):
        res = create_results(testdata / 'psn' / 'frem_dir1', method='bipp')
    ofv = res.ofv['ofv']
    assert len(ofv) == 5


def test_modeling_create_results(testdata):
    with pytest.warns(UserWarning):
        res = tools.create_results(testdata / 'psn' / 'frem_dir1', method='bipp')
    ofv = res.ofv['ofv']
    assert len(ofv) == 5


def test_create_report(testdata, tmp_path):
    res = tools.read_results(testdata / 'frem' / 'results.json')
    shutil.copy(testdata / 'frem' / 'results.json', tmp_path)
    tools.create_report(res, tmp_path)
    html = tmp_path / 'results.html'
    assert html.is_file()
    assert html.stat().st_size > 500000
