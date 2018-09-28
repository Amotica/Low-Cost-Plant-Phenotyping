#   ================================
#   CHANGE THIS FOR EACH EXPERIMENT
#   ================================
home_dir = "/home/pszjka/pixel_wise/"
data_dir = "/mnt/beast/db1/pszjka/"

#home_dir = ""
#data_dir = ""

#   flowers_fore_back / flowers_multi_class/ leaf / CamVid ****
dataset = "flowers_fore_back"
#   ===============================

npy_data_path = data_dir + dataset + '/'

train_data_file = data_dir + dataset + '/train.txt'
val_data_file = data_dir + dataset + '/val.txt'
test_data_file = data_dir + dataset + '/test.txt'

train_data_file_gen = data_dir + dataset + '/train/'
val_data_file_gen = data_dir + dataset + '/val/'
#val_data_file_gen = data_dir + dataset + '/A3/'
test_data_file_gen = data_dir + dataset + '/test/'

mask_train_data_file_gen = data_dir + dataset + '/trainannot/'
mask_val_data_file_gen = data_dir + dataset + '/valannot/'
#mask_val_data_file_gen = data_dir + dataset + '/A3annot/'
mask_test_data_file_gen = data_dir + dataset + '/testannot/'


#   ================================
#   CHANGE THIS FOR EACH EXPERIMENT
#   ================================
# segnet_basic / fcn / squeezed_fcn / squeezed_segnet / tiny_segnet_depool / segnet_depool ****
model_type = "squeezed_segnet"

misc_dir = data_dir + 'misc/' + dataset + '/' + model_type
misc_dir_eval = data_dir + 'misc/' + dataset + '/evaluate/' + model_type
num_epoch = 200
batch_size = 6
# leaf=3, flowers_fore_back=2, flowers_multi_class=13, CamVid=12 ****
num_classes = 2

data_gen = False
droput_rates = [0.0, 0.0, 0.0, 0.0, 0.0]
weights = False   # ******************

'''
leaf = [0, 1.50128114, 1.0]      
flowers_fore_back = [0, 1.81584014]
flowers_multiclass = [0, 1.16598638, 3.84556845, 1.39737896, 0.78872808, 1.6174536, 1.20223472, 0.76162298, 1.02294366,
                   0.89280318, 0.88975886, 1.0, 0.8287119] 
camvid = [0.28852256, 0.10198304, 4.69145372, 0.09144813, 0.30366565, 0.1620776, 2.96745203, 0.86021236,
                   1.51650553, 4.07028477, 1.19403504, 0]
'''
class_weighting = [0, 1.81584014]  # flowers_multiclass ******************

#   image related parameters
channels = 3
img_cols = 224  # 160 or 224 or 96   ******************
img_rows = 224  # 120 or 224 0r 320  ******************