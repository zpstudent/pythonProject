import os.path
import re
import rawpy
import imageio
import numpy as np
import glob
# .ARW为索尼Sony相机RAW格式
# .CR2为佳能canon相机RAW格式

load_sources = input('请输入读取的图片的文件夹的绝对路径：')
load_save = input('请输入保存图片的文件夹的绝对路径：')
load_save = load_save.replace('"', '')
load_sources = load_sources.replace('"', '')
select = input('请选择要读取图片的类型：1.CR2  2.RAW(填写相应序号即可）：')
print('正在转换......')
if int(select) == 1:
    raw_imags = glob.glob(os.path.join(load_sources, '*.CR2'))
else:
    raw_imags = glob.glob(os.path.join(load_sources, '*.RAW'))

n = 0
for raw_imag in raw_imags:

    raw = rawpy.imread(raw_imag)

# use_camera_wb 是否执行自动白平衡，如果不执行白平衡，一般图像会偏色
# half_size 是否图像减半
# no_auto_bright 不自动调整亮度
# output_bps bit数据， 8或16
    img = raw.postprocess(
        use_camera_wb=True,
        half_size=False,
        no_auto_bright=False,
        output_bps=16)
    img = np.float32(img / (2**16 - 1) * 255.0)
    img = np.asarray(img, np.uint8)
    pattern = re.compile(r'([^<>/\\\|:""\*\?]+)\.\w+$')
    data = pattern.findall(raw_imag)
    data_now = data[0]
    Newdir = os.path.join(load_save, str(data_now) + '.jpeg')
    imageio.imsave(Newdir, img)
    n = n + 1
print(f'转换成功！一共转换了{n}张图片。')
