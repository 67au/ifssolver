echo "[intel_map]" > config.ini
echo "COOKIES=$COOKIES" >> config.ini
echo "LAT=$LAT" >> config.ini
echo "LNG=$LNG" >>config.ini
echo "RADIUS=$RADIUS" >> config.ini
echo "[common]" >> config.ini
echo "TEMP_DIR=/ifs-tmp" >>config.ini
echo "OUTPUT_DIR=/ifs-output" >> config.ini
echo "[ifs]" >>config.ini
echo "IFS_IMAGE=/ifs.jpg" >>config.ini
echo "COLUMN=$COLUMN" >>config.ini
echo "[proxy]">>config.ini
echo "enable = False">>config.ini
echo "url = socks5://127.0.0.1:1080">>config.ini
echo "[silx]">>config.ini
echo "devicetype = all" >>config.ini
echo "platformid = 0" >>config.ini
echo ";deviceid = 0" >>config.ini
python3 ifssolver.py --auto
