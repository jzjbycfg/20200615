# 进行测试的主程序
from utils import *

# 加载训练好的模型
location_model_path = ''
classification_model_path = ''
location_model = load_model(location_model_path)
classification_model = load_model(classification_model_path)

# 加载数据，还是处理成data_loader的形式
test_path = ''
test_data = [os.path.join(test_path, fn) for fn in os.listdir(test_path)]  # 获取每一个study的绝对路径

# inference
results = []  # 把每一个样例的结果放在result这个list中，最后再统一处理
for i, study_dir in enumerate(test_data):
    data, data_info = load_data(study_dir)
    # 目标定位模型，获取目标patches
    patches = location_model.predict(data)
    # 目标分类模型，输入目标patches，获取每个目标的分类标签
    tags = classification_model.predict(patches)
    res = wrap(tags, data_info)  # 把预测的结果打包为提交的格式
    results.append(res)

# 保存结果
filename = r'C:\Users\Administrator\Desktop\2020spring\projectsLJN\jzjbycfg\tmp\ans.json'
with open(filename, 'w') as file_obj:
    json.dump(results, file_obj)

