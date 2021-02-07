# IFS Script

## 环境要求

- Python 3.9 及以上

```shell
pip install -r requirements.txt
```

## 用法

将 sample_config.ini 复制一份为 config.ini ，修改后再运行 ifssolver.py

### 显示帮助

```
> python ifsscript.py -h
usage: ifsscript.py [-h] [--config filename] [-u | -d | -b] [--spilt | --solve]

A script used to obtain passcode from special picture

optional arguments:
  -h, --help         show this help message and exit
  --config filename  Configure file, default = 'config.ini'
  -u, --update       Update portals metadata only
  -d, --download     Download portals data for metadata only
  -b, --both         Both update and download
  --spilt            Spilt picture only
  --solve            Get passcode from picture automatically
```

### 指定配置文件

```shell
python ifsscript.py --config test.ini
```

### 下载地图 Portal 数据

```shell
python ifsscript.py --update
```

### 下载 Portal 照片

```shell
python ifsscript.py --download
```

### 仅分割图像

```shell
python ifsscript.py --spilt
```

### 处理结果

```shell
python ifsscript.py --solve
```

## 注意事项

- 脚本采用异步编程，相比原版本速度会更快
- 采用tqdm显示程序运行的进度 
- config.ini 配置文件按提示文件里面的注释来修改
- 更新了分割图像的方案，对于绝大部分的图像来说不用修改设定的阈值。
  如果出现图像连接在一起，则增大阈值；出现图像分裂，则缩小阈值
- 需要先更新数据，再下载portal图像，最后才匹配，其中哪一步错了也可以重试
- 忘记把代理的配置写上了，有机会再加上
