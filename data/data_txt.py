import os

# 设定你的文件夹路径
folder_path = './garment'

# 创建或打开一个txt文件
with open('garment.txt', 'w') as file:
    # 遍历文件夹中的所有文件和子文件夹
    for filename in os.listdir(folder_path):
        # 只处理文件，忽略子文件夹
        if os.path.isfile(os.path.join(folder_path, filename)):
            # 写入文件名到txt文件，每个文件名一行
            file.write(filename + '\n')
