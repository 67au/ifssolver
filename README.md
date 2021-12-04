# ifssolver

用于解决每月都要来一次的问题

## 环境要求

- python >= 3.7
- opencv-python >= 4.5.3
- aiofiles >= 0.7.0
- httpx >= 0.19.0
- httpx-socks >= 0.4.1
- tqdm >= 4.62.3
- silx >= 0.15.2 [optional]
- pyopencl >= 2021.2.8 [optional]

> 如果需要使用 GPU 加速的 sift 算法，请安装上面可选依赖和显卡对应的驱动，并使用 [clinfo](https://github.com/Oblomov/clinfo) 等工具检查 opencl 的状态，确保可用。安装 pyopencl 的方式可以参考 [这里](https://documen.tician.de/pyopencl/misc.html#installation) ，在 Windows 上推荐使用 [预编译二进制包](https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyopencl)

## 使用方法

```
> python3 ifssolver.py --help
usage: ifssolver.py [-h] [--config filename]
                    [[--download-csv | --download-img | --download-all |]
                    --split | --draw | --auto] [--method opencv] [--no-clean]

ifssolver

optional arguments:
  -h, --help         show this help message and exit
  --config filename  Configure file, default = 'config.ini'
  --download-csv     download metadata csv only
  --download-img     download image for metadata csv
  --download-all     download image after updating metadata csv
  --split            split ifs picture only
  --draw             draw result only
  --auto             AUTO
  --method opencv    sift algorithm provider, opencv or silx
  --no-clean         use the last files
```

### 从 GitHub 上下载该项目

```shell
git clone --recurse-submodules https://github.com/67au/ifssolver.git
```

### 创建配置文件

参考提供的 config.sample.ini ，使用 --config 参数指定配置文件，默认： 'config.ini'

### 下载地图数据

下载地图指定区域的元数据

```shell
python3 ifssolver.py --download-csv
```

下载元数据内包含的照片

```shell
python3 ifssolver.py --download-img
```

顺序执行上面两步

```shell
python3 ifssolver.py --download-all
```

### 匹配图像

匹配完成后输出标记的 output.jpg 和排序后的 result.csv

```shell
python3 ifssolver.py --split
```

### 生成图像

依据 result.csv 生成 passcode.jpg

```shell
python3 ifssolver.py --draw
```

### AUTO

顺序完成 下载地图数据、匹配图像、生成图像 这三步

```shell
python3 ifssolver.py --auto
```

### 可选参数

- method: 指定匹配用的方法，参数：opencv 或 silx，默认为 opencv。
  - opencv: opencv-python 中的 sift 
  - silx： silx-kit 项目中 GPU 加速的 sift 
- no_clean: 指定该参数运行时，不会覆盖已存在的照片和匹配计算的缓存。因为不同方法之间的缓存不通用，所以更换后应禁用该参数运行以覆盖缓存

## 注意事项

- 只有下载地图元数据部分需要 cookies
- 下载数据这部分使用代理的话，会被限制为单线程下载（详情请查看 [IntelMapClient](https://github.com/67au/intel-map-client) ），推荐使用使用命令行代理的方式（例如 proxychains ）
- ~~silx-kit 提供的算法匹配关键点数相比 opencv 提供的少，只影响 ifs 图像上缩放较小的照片，造成无法识别。（可以通过修改参数解决，但参考相关资料）~~ [[查看 commit]](https://github.com/67au/ifssolver/commit/b93b53969edc8dd55147f687c27e6c992eb4ad72)
- 使用 silx-kit 需要安装 pyopencl，指定使用 GPU 计算，~~但是对于多显卡没有将接口暴露所以没法指定显卡，需要修改 solver/extensions/sift_silx.py~~ [[查看 commit]](https://github.com/67au/ifssolver/commit/6e690eb4087d91acc4dbcccea41fdc4b0f25366b)
- opencv 提供的算法是基于 CPU 计算

## 使用 Docker 執行

- Build Docker image，建立對應目錄:

```shell
docker build . -t ifssolver:latest
mkdir /tmp/ifs-tmp
mkdir /tmp/ifs-output
```

- 將影像儲存在 `/tmp/ifs.jpg` (可自訂，掛載至/ifs.jpg即可)

- 複製 `.env.example` 為 `.env` 檔案，編輯 `.env` 中相對應參數

- 執行

```shell
docker run --rm -v /tmp/ifs-tmp:/ifs-tmp -v /tmp/ifs-output:/ifs-output -v /tmp/ifs.jpg:/ifs.jpg --env-file ./.env -it ifs
```

- 即可在 `/tmp/ifs-output` 取得結果

## 致谢

- [@ROrdo](https://github.com/RogerRordo) 
- [UESTC-Ingress/IFSolver](https://github.com/UESTC-Ingress/IFSolver)
- [pierrepaleo/sift_pyocl](https://github.com/pierrepaleo/sift_pyocl)
- [opencv: Introduction to SIFT (Scale-Invariant Feature Transform)](https://docs.opencv.org/4.5.4/da/df5/tutorial_py_sift_intro.html)
- [opencv: Feature Matching + Homography to find Objects](https://docs.opencv.org/4.5.4/d1/de0/tutorial_py_feature_homography.html)
