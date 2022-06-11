from configparser import ConfigParser
from pathlib import Path

from .types import PathType


class ConfigProxy:

    def __init__(self, config: ConfigParser):
        self._config = config
        self.cookies = self._config.get('intel_map', 'COOKIES', raw=True, fallback=None)
        self.lat = self._config.getfloat('intel_map', 'LAT', fallback=None)
        self.lng = self._config.getfloat('intel_map', 'LNG', fallback=None)
        self.radius = self._config.getint('intel_map', 'RADIUS', fallback=None)
        self.temp_dir = Path(self._config.get('common', 'TEMP_DIR'))
        self.output_dir = Path(self._config.get('common', 'OUTPUT_DIR'))
        self.ifs_image_path = Path(self._config.get('ifs', 'IFS_IMAGE'))
        self.column = self._config.getint('ifs', 'COLUMN')
        self.proxy = self._config.get('proxy', 'url') \
            if self._config.getboolean('proxy', 'enable', fallback=False) else None
        self._prepare_and_check()

    @property
    def silx(self) -> dict:
        return dict(
            devicetype=self._config.get('silx', 'devicetype', fallback='all'),
            platformid=self._config.getint('silx', 'platformid', fallback=None),
            deviceid=self._config.getint('silx', 'deviceid', fallback=None)
        )

    @property
    def portal_images_dir(self) -> Path:
        return self.temp_dir.joinpath('images')

    @property
    def portal_features_dir(self) -> Path:
        return self.temp_dir.joinpath('features')

    @property
    def output_sub_dir(self) -> Path:
        return self.output_dir.joinpath(self.ifs_image_path.stem)

    @property
    def download_errors_txt(self) -> Path:
        return self.output_sub_dir.joinpath('download_errors.txt')

    @property
    def split_errors_txt(self) -> Path:
        return self.output_sub_dir.joinpath('split_errors.txt')

    @property
    def metadata_csv(self) -> Path:
        return self.output_sub_dir.joinpath('metadata.csv')

    @property
    def match_result_csv(self) -> Path:
        return self.output_sub_dir.joinpath('match_result.csv')

    @property
    def match_result_jpg(self) -> Path:
        return self.output_sub_dir.joinpath('match_result.jpg')

    @property
    def passcode_jpg(self) -> Path:
        return self.output_sub_dir.joinpath('passcode.jpg')

    def _prepare_and_check(self):
        # check
        if not self.ifs_image_path.exists():
            raise FileNotFoundError(f'IFS 图像 ({str(self.ifs_image_path)}) 不存在')

        # prepare for mkdir
        self.portal_images_dir.mkdir(parents=True, exist_ok=True)
        self.portal_features_dir.mkdir(parents=True, exist_ok=True)
        self.output_sub_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def load_config(cls, config_path: PathType) -> 'ConfigProxy':
        config = ConfigParser()
        config.read(config_path, encoding='utf-8')
        return cls(config)