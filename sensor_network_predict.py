from random import seed
from mlp_mestrado import neural_network_m2 as nn

seed(1)

network = [
    [
        {'weights': [2.3121229997496671, -0.015867355115746613, -2.8319910803240433, 1.9107824945515648,
                     -2.2217169121631954, 4.9348145411707511, 2.2300608657405872, 3.1300739142221676,
                     1.8163491027870931, 2.0479950214781244, 0.62541375203412519, 13.886391101809842,
                     -24.771880592531978, -13.567324140086871, -0.62225335746811394, 0.042014974145311544,
                     11.301786621481055, 1.5959511792729701, -7.689460420397122, 31.340192620064752,
                     5.955093584729461, -3.7133810194269055, 0.14279798855539289, -1.0827059066425557,
                     3.4168718662786093]},
        {'weights': [6.3870378118648121,
                     -2.7732151053790717,
                     2.3811579794327247,
                     20.492393332056565,
                     2.3736807766045915,
                     -4.7200222007469694,
                     -2.4271383006888279,
                     3.3259679598200362,
                     -2.7597742763053126,
                     9.2558302541757502,
                     -3.4280072784206119,
                     6.3731602035499693,
                     9.8271775981066529,
                     11.045612503498777,
                     -4.9263403617890882,
                     -1.7868407580681533,
                     -0.30096825204707223,
                     -1.8771490158105959,
                     -12.369258128640404,
                     5.8304246782168923,
                     -11.579350814674161,
                     0.44022065049291464,
                     -0.46646659678306673,
                     59.055155669735157,
                     -16.054406933240415]},
        {'weights': [0.01021248887624656, 5.2047065107592827, 0.59793758437951194, -0.68863534275832949,
                     -1.6661892055086835, -6.7956811145588709, -0.57892055766037498, -0.041474969198120569,
                     2.1471741393873245, -1.0053113027245082, 2.3717382310617254, 1.9826739880741435,
                     16.760127531579933, 1.8416139087889196, -0.11221964731173871, -1.8904910863839137,
                     -48.778435343482229, 0.6637763185030825, -19.250413864232538, -2.7458224918873988,
                     -0.11832685833190432, 1.162138097146554, -0.84978669400467588, -0.56164293381208574,
                     6.3520088637851231]},
        {'weights': [1.0836791184296115,
                     -3.3275319149593772,
                     2.60833287038691,
                     0.38861198571888217,
                     20.045948039129701,
                     -3.3874863999352454,
                     -0.87191476658432376,
                     -11.620676642307643,
                     4.5624940076838429,
                     7.7232112040592851,
                     -7.0803803959478309,
                     -3.0771321946090602,
                     -2.5511666793817365,
                     -28.522564027148249,
                     -6.0869406679298983,
                     0.76819934126733724,
                     -4.1595109027964474,
                     20.171487178152162,
                     8.7406435418492148,
                     -2.3070653849276481,
                     29.506521401884239,
                     -5.0340205415249235,
                     -2.6358923937965466,
                     0.21610203163612615,
                     15.014449752962593]},
        {'weights': [0.34904240605351883, 0.50099192871512621, -1.4705919153072042, 0.033359470590090326,
                     -0.63534340484831886, 0.11608208102315323, -0.52453601132370942, -0.0037979593654979883,
                     -0.734681887665942, -0.1086112969244665, 0.54858373237176339, -0.84383819551092909,
                     -1.0518741763415105, 3.3173775909524523, -181.54577300713422, -0.8630277127599737,
                     4.0366631974526381, 3.6339254095561246, 0.69703145463884686, 0.50294030059196704,
                     1.5422729601829375, -0.53503394590844855, 0.17312036163177194, 0.50337489039861216,
                     16.050708263976844]},
        {'weights': [-0.86177373211771213,
                     0.24302678075484876,
                     -1.8474103669512363,
                     1.2064767273491521,
                     -1.6893088215252456,
                     -0.00520689256287737,
                     0.37739059155939325,
                     0.074842592960160645,
                     0.92360771162288824,
                     -0.094232976341800043,
                     -0.0021367310566354717,
                     -0.22343771901956638,
                     0.14832380330751105,
                     -3.4123948166249218,
                     0.24582702755612446,
                     0.24600490897396377,
                     -7.057942662346016,
                     -20.86823686376141,
                     43.127385357803632,
                     109.44715542616554,
                     1.3867367915427358,
                     0.67534273201141415,
                     2.3459719773636389,
                     3.3905886506324214,
                     -6.0354279775479185]},
        {'weights': [2.7232391427543448, -3.1384862121092976, 0.78084802528022168, 0.052462363464042414,
                     2.8381181458414999, 3.620419990110888, -1.4598753607404633, 0.46536419959420888,
                     0.83791453336884048, 0.096539211315155221, -0.36686069209217853, 44.831311848106637,
                     0.80617658893361166, 7.9326962308749662, -1.5632013681421557, 2.3946202822801013,
                     8.424831381853183, 84.214863590452111, 0.66503932447013225, 3.6130431056255401,
                     -4.4832161897340796, 1.4530875456859604, 0.79496788832370524, -5.1982819100101345,
                     -14.341229481471823]}],
    [
        {
            'weights': [33.65270807879849, -5.9254154981007385, -39.908638219933657, -30.581025123665007,
                        -7.5756151748906095, 28.62737757597456, -1.0337596326696712, -22.764172082357756],
        }, {'weights': [-22.689055981100228, 6.5429498972052462,
                        -21.696092020500647, -2.9020499766164556,
                        -14.204651303859798, 16.479119212878661,
                        -3.1169520192462583, 6.9247541122546732]},
        {'weights': [3.7544226600841419,
                     -0.7550432465927408,
                     1.394044814376024,
                     -24.377883222815285,
                     -34.304472147267312,
                     0.38532453390007426,
                     -0.99121818413824048,
                     26.831839560061731]},
        {'weights': [-8.2101326380023423, -8.1382708768019754, -21.757694517028323, -14.458779098157221,
                     -14.425665687514677, 8.0005695861084956, -5.4880062492701764, 28.45307902992397]},
        {'weights': [2.4341246294329975, -5.4284789487332983,
                     19.923977525097566,
                     21.829342339951204, 8.8001310001696869,
                     5.1777023400414013,
                     -1.944083758297489, 0.24286791825977985]},
        {'weights': [30.792150911615799, 15.175137842107391, 5.569341569524199, -27.45560962179648,
                     1.1209335771315441, -34.863263929976441, -32.222221135520179, 16.254252727317912]},
        {'weights': [-5.087865513541515, -0.57599714213067232,
                     -1.5383707610855126, -15.688921136036672,
                     -10.05309169527942, 9.8035298378164093,
                     -30.975714432296733, 22.538288842093628],
         }
    ],
    [
        {'weights': [-27.271225494232731, -28.862184474577496, 33.047952728950456, 10.560177709006398,
                     3.208789209462557, -32.71437227633929, -8.3537560897570593, -8.3370902734705119]},
        {'weights': [13.806131928410451, 14.89282166175291, -30.429080253626289,
                     -4.4089353468932737, 13.690174157418731, 19.164675880422099,
                     14.86441220407214, -9.5610319524096568]},
        {'weights': [-17.207987215261308,
                     11.542351239853469,
                     12.535066096685807,
                     -15.065564881715995,
                     -28.138758517494779,
                     32.583266302272413,
                     -10.926179853202271,
                     -12.10476312700429]},
        {'weights': [31.897733444261203, 17.380718186599552, 18.544969793784155, -25.349002354753196,
                     -18.178662315603127, -14.177806501734446, -13.75187668387739, -17.417950680962793]}
    ]
]

filename_test = 'sensor_readings_24_test.csv'
dataset_test = nn.load_csv(filename_test)
for i in range(len(dataset_test[0]) - 1):
    nn.str_column_to_float(dataset_test, i)

# convertendo coluna de classe para inteiros
lookup = nn.str_column_to_int(dataset_test, len(dataset_test[0]) - 1)
# normalizando dados
minmax = nn.dataset_minmax(dataset_test)
nn.normalize_dataset(dataset_test, minmax)

for row in dataset_test:
    prediction = nn.predict(network, row)
    print('Expected=%d, Got=%d' % (row[-1], prediction))
