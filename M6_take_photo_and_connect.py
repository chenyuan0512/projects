

import subprocess
import os
import time


def findCanon():
    """
    藉由有連接時wsl lsusb 會輸出 => Bus 001 Device 004: ID 04a9:32e7 Canon, Inc. Canon Digital Camera
    如果找到會return True
    """
    cmdout = subprocess.run(
        ["wsl", "lsusb"], capture_output=True, text=True, encoding="utf-8")
    text = cmdout.stdout
    index_output = text.find("Canon Digital Camera")

    if index_output != -1:
        return True
    else:

        return False


def connect_camera():
    try:
        subprocess.run(["usbipd", "bind", "--busid", "1-4"], check=True)
        subprocess.run(["usbipd", "attach", "--wsl",
                       "--busid", "1-4"], check=True)

        return True

    except Exception as e:
        """
        你可能usb沒插好 電源正在休眠 或是你沒關 或是你沒關 EOS Utility 最後有可能就是你用的相機名稱電腦看上去沒有Canon這個字眼
        """
        print("please check whether the camera is connected to your device or sleeping or power off !!! ")
        print("or you forget to close the app 'EOS Utility' !!!")
        print(f"錯誤: {e}")

        return False


def get_camera_port(camera_keyword):
    # 取得 gphoto2 相機列表
    result = subprocess.run(
        ["wsl", "gphoto2", "--auto-detect"], capture_output=True, text=True)
    lines = result.stdout.strip().split('\n')
    print(lines)

    for line in lines:
        if camera_keyword in line:
            parts = line.split()
            port = parts[-1]  # 最後一個是 port
            print(f"找到 {camera_keyword} 對應的 port: {port}")
            return port

    raise ValueError(
        f"Can`t find the camera include the keyword => '{camera_keyword}'")


def capture_image(path, filename="photo.jpg", camera_keyword=None):
    prefix_path = "//mnt//"
    filepath = prefix_path + path + "//" + filename

    if camera_keyword is not None:
        port = get_camera_port(camera_keyword)
        cmd = ["wsl", "gphoto2", "--port", port,
               "--capture-image-and-download", "--filename", filepath]
    else:
        cmd = ["wsl", "gphoto2", "--capture-image-and-download",
               "--filename", filepath]

    # 執行拍攝
    subprocess.run(cmd)
    time.sleep(2)

    return filepath


def pathtest(path, filename):
    prefix_path = "//mnt//c"
    filepath = os.path.join(prefix_path, path, filename)
    print(filepath)


def printwd():
    # correct_path = "/mnt/c/Users/Sulab/Desktop/Program_YiTingVersion/python_findspotpositoin/python/test"
    # subprocess.run(["wsl", "cd",correct_path])
    result = subprocess.run(["wsl", "pwd"], capture_output=True, text=True)
    # result = subprocess.run(["wsl", "pwd"], capture_output=True, text=True)
    # print(result.stdout.strip())
    return result.stdout.strip()  # 去掉換行符號


def check_connection():
    if findCanon() == False:
        print("相機未連接")
        connect_camera()
        time.sleep(2)

        if findCanon() == False:
            print("無法連接相機，請檢查相機連接狀態")
            return False
        else:
            print("相機已連接")
            return True
    else:
        print("相機已連接")
        return True


def main():
    # path 參數的開頭若有 /，導致它被當作絕對路徑，而忽略了 prefix_path
    # pathtest("Users/Sulab/Desktop/Program_YiTingVersion/python_findspotpositoin/python/test","test.jpg")
    # print(findCanon())

    check_connection()

    now_str = time.strftime("%Y%m%d_%H%M%S")  # 取得現在的時分
    file_path = "c//Users//Sulab//Desktop//Program_ChenYuan_Version//photo_from_M6"
    capture_image(file_path, f"{now_str}.jpg", "Mark")
    time.sleep(2)

    return "photo captured successfully"
