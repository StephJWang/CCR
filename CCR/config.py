import glob
# reduced_lp_distance_test

# path = '/Users/junwang/Documents/Social Choice/data/manual'
# filename = 'table1.soi'

path = '/Users/junwang/Documents/Social Choice/data/M10N10000-soi-Mode2'
filename = '*.soi'
# filename = 'M10N10000-000901.soi'

result_path = '/Users/junwang/Documents/PycharmProjects/CCR/results/'

dist_path = '/Users/junwang/Documents/PycharmProjects/CCR/results/M8N10k-soi-Mode3-dist/'
# dist_result = result_path + 'M8N10k-soi-Mode3-1k-dist.txt'
dist_result = result_path + 'rf-M8N10k-soi3-s1k-2k_4k-dist.json-1001-1300.random_forest'

dict_pickle = result_path + 'rf-M8N10k-soi3-1001-1300-random_forest.pickle'

dist_folder = '/Users/junwang/Documents/PycharmProjects/MOVRP/results'
distfile = 'M8N10000-01*.soi-dist.txt'
# distfile = 'M8N10000-01*.soi-dist_6.txt'

write_distfile = 'M5N10ksoc3-100k-dist.txt'

# dist_folder = '/Users/junwang/Documents/PycharmProjects/MOVSTV/results'
# distfile = 'report-random_forest-M8N10k-soi3-all1101_1170-distpred_6.txt'

# path = '/Users/junwang/Documents/Social Choice/data/M9N10000-soc-Mode2'
# filename = 'M9N10000-001.soc'
# path = '/Users/junwang/Documents/Social Choice/data/M10N10000-soc-Mode2'
# filename = 'M10N10000-001.soc'

# path = '/Users/junwang/Documents/Social Choice/data/M7N10000-soi-Mode2'
# filename = 'M7N10000-00005.soi'
# filename = 'ED-00007-00000059.soi'

results_folder = '/Users/junwang/Documents/PycharmProjects/CCR/results'
# results_folder = '/Users/junwang/Documents/PycharmProjects/MOVSTV/M8N1k-results-20191018/'
results_filename = 'margin_*.txt'

test_results = 'results-MBP-ML2-M16N10k-k8-20220307_1k.txt'
curvedump = 'results-MBP-ML2-M16N10k-k8-20220308_1k.pickle'
curvedumps = 'results2020*_66profiles.pickle'

curvedump1 = 'results20200422_dfslp.pickle'
curvedump2 = 'results20200417_dfslp.pickle'

# data_folder = '/Users/junwang/Documents/PycharmProjects/MOVSTV/NSW2015/'
# data_folder = '/Users/junwang/Documents/Social Choice/data/M8N10000-soc-Mode3'
data_folder = '/Users/junwang/Documents/Social Choice/data/M4N20-soc-Mode3' # ------------------------------ for MOV
# data_folder = '/Users/junwang/Documents/Social Choice/data/M6N100-soc-Mode3'
# data_folder = '/Users/junwang/Documents/Social Choice/data/M8N1000-soc-Mode3'
# data_folder = '/Users/junwang/Documents/Social Choice/data/M16N10000-soc-Mode3'
# data_folder = '/Users/junwang/Documents/Social Choice/data/manual'
# data_folder = '/Users/junwang/Documents/Social Choice/data'
# data_folder = '/Users/junwang/Documents/Social Choice/data/M5N10000-soc-Mode2'
# data_folder = '/Users/junwang/Documents/Social Choice/data/M5N10000-soc-Mode2'
# data_folder = '/Users/junwang/Documents/Social Choice/data/M6N24-soc-Mode3'
# data_folder = '/Users/junwang/Documents/Social Choice/data/M9N10000-soc-Mode2'



write_folder = '/Users/junwang/Documents/Social Choice/data/M8N10000-soi-Mode3-blom/'
# data_folder = '/Users/junwang/Documents/Social Choice/data/manual'
# data_filename = 'M10N10000-000901.soi'
# data_filename = 'ccr.soc'
# data_filename = 'table1.soi'
data_filename = '*.soc'

fig_folder = '/Users/junwang/Documents/PycharmProjects/MOVSTV/figures/'
json_folder = '/Users/junwang/Documents/PycharmProjects/MOVSTV/json/'


ml_path = '/Users/junwang/Documents/PycharmProjects/MOVSTV/results/'
# ml_result = 'M8N10000-soi-Mode3-blom_M8N10000-00355_distance.txt'
ml_result = 'M8N10k-soi-Mode3-blom_M8N10k-371_distance.txt'
# filename = glob.glob('*.soi')

# -----------------------------for machine learning-----------------------------------
# for feature.py
profile_folder = '/Users/junwang/Documents/Social Choice/data/M8N10000-soi-Mode3'
profile_filenames = '*.soi'

dist_result2 = result_path + 'M5N10k-soc3-5k-experiment-20200522.txt'
pickledump = result_path + 'M5N10k-soc3-5k-experiment-20200522.pickle'

# for main() in features.py
featuresinput = result_path + 'M16N10k-soc3-100k-experiment-20220223.pickle'
featuresoutputprefix = result_path + 'M16N10k-soc3-100k-experiment-20220223'

# for ml.py
mlinputfile2 = result_path + 'features-M5N10k-soc3-10k-dist-mac-*.json'
mloutputfile = result_path + 'test.json'

mlinputfile = result_path + 'M16N10k-soc3-100k-experiment-20220223.json'

mltrain = result_path + 'features-M8N10k-soi3-all1001_2000-dist_1.json'
# mltrain = result_path + 'features-M10N10k-soi2-1M-0.1_1M-dist_8-thinkpad.json'
mltest = result_path + 'features-M8N10k-soi3-all1001_2000-dist_1.json'
# mltest = result_path + 'features-M10N10k-soi2-1M-1M_1.1M-dist_8-desktop.json'

# model_name = result_path + 'report-random_forest-M8N10k-soi-Mode3-smp1k-2k_4k-dist.pkl'  # TEST MSE=510k, the best so far

write_model_name = result_path + 'test.pkl'

models_path = '/Users/junwang/Documents/PycharmProjects/CCR/models/'
model_name = models_path  + 'model-nn-M5N10ksoc3-100k-mindist-20200511-test_60000-69999.pkl'
model_name_prefix = 'model-krr-M16N10ksoc3-100k-20220224'
