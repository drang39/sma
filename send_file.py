import requests
import time
import os
from PIL import Image



def send_file(fn):
    img_path = fn
    with open(img_path,'rb') as f:
        img = f.read()
    datas={'img':img}
    st = time.time()
    url_root='http://127.0.0.1:5000/'
    responde = requests.post(url_root+'predict_file',files=datas)
    print((time.time()-st))
    print(responde.text)



if __name__ == "__main__":
    start_time = time.time()
    # c=0
    for img in os.scandir('img_test'):
        if '.bmp' in img.name:
            # c+=1
            fn = img.path
            send_file(fn)
            # if c==359:
            #     pass
    t=time.time()-start_time
    print(t)
    print(t/360)
    # for i in range(100):
    #     send_file('1.bmp')