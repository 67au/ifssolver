# ifssolver

用于解决一个月都要来一次的问题

## Requirements
- `python>=3.7`
- `aiofiles~=0.8.0`
- `httpx~=0.23.0`
- `httpx-socks[asyncio]~=0.7.4`
- `intel-map-client==0.2.1`
- `opencv-python~=4.5.5.64`
- `tqdm~=4.64.0`

- `pyopencl==2022.1.5 [optional]`
- `silx==1.0.0 [optional]`

> 如果需要使用 GPU 加速的 sift 算法，请安装上面可选依赖和显卡对应的驱动，并使用 [clinfo](https://github.com/Oblomov/clinfo) 等工具检查 opencl 的状态，确保可用。安装 pyopencl 的方式可以参考 [这里](https://documen.tician.de/pyopencl/misc.html#installation) 。

## Installation

从 GitHub 上下载该项目

```shell
git clone https://github.com/67au/ifssolver.git
```

### 安装依赖

推荐使用虚拟环境 [venv](https://docs.python.org/zh-cn/3/library/venv.html)

```shell
pip install -r requirements.txt
```

或者安装可选依赖

```shell
pip install -r requirements-silx.txt
```

## Usage

```
$ python3 ifssolver.py --help
usage: ifssolver.py [-h] [--config filename]
                    [--download-csv | --download-img | --download-all]
                    [--split] [--draw] [--auto] [--metadata METADATA]
                    [--method opencv] [--no-clean]

ifssolver

optional arguments:
  -h, --help           show this help message and exit
  --config filename    Configure File, default = 'config.ini'
  --download-csv       download metadata
  --download-img       download image by metadata
  --download-all       download image after updating metadata
  --auto               AUTO
  --metadata METADATA  use specified METADATA
  --method opencv      sift algorithm provider, opencv or silx
  --no-clean           no clean cache file
```

### 创建配置文件

参考提供的`config.sample.ini`，使用`--config`参数指定配置文件，默认：`config.ini`

### 下载地图数据

下载地图的元数据（需要Cookies）

```shell
python3 ifssolver.py --download-csv
```

下载元数据中包含的照片

```shell
python3 ifssolver.py --download-img
```

顺序执行上面两步

```shell
python3 ifssolver.py --download-all
```

### 识别图像

识别结果输出到指定目录 `<OUTPUT_DIR>`

```shell
python3 ifssolver.py --split
```

### 生成 Passcode

```shell
python3 ifssolver.py --draw
```

### AUTO

```shell
python3 ifssolver.py --auto
```

### 可选参数
- `--method`: 指定匹配用的方法，参数：opencv 或 silx，默认为 opencv。
  - `opencv`: opencv-python 中的 sift 
  - `silx`： silx-kit 项目中支持 GPU 加速的 sift 
- `--no-clean`: 默认禁用，使用该参数可以跳过覆盖缓存文件。
- `--metadata`: 指定`METADATA` csv 文件以代替利用 Cookies 从 IntelMap 上下载的数据

## Note

- 只有下载地图元数据部分需要 Cookies
- `--meatadata`参数只兼容 [IITC-Ingress-Portal-CSV-Export](https://github.com/Zetaphor/IITC-Ingress-Portal-CSV-Export) 这个插件

## Credit

- [@ROrdo](https://github.com/RogerRordo) 
- [UESTC-Ingress/IFSolver](https://github.com/UESTC-Ingress/IFSolver)
- [pierrepaleo/sift_pyocl](https://github.com/pierrepaleo/sift_pyocl)
- [opencv: Introduction to SIFT (Scale-Invariant Feature Transform)](https://docs.opencv.org/4.5.4/da/df5/tutorial_py_sift_intro.html)
- [opencv: Feature Matching + Homography to find Objects](https://docs.opencv.org/4.5.4/d1/de0/tutorial_py_feature_homography.html)

## License

MIT License
