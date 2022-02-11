import numpy as np
import cv2
from keras.preprocessing import image
import sys
import datetime
import telepot
from telepot.loop import MessageLoop
from time import sleep

#-----------------------------
now = datetime.datetime.now() # Mendapatkan tanggal dan waktu
#mengaktifkan telegram bot
bot = telepot.Bot('1861704341:AAEO6xT3Q9xAAtnsPPSnFPycgdIwQVdRGDU')
chat_id=-464454490
print (bot.getMe())

#inisialisasi kamera
def gstreamer_pipeline(
    capture_width=1280,
    capture_height=720,
    display_width=1280,
    display_height=720,
    framerate=60,
    flip_method=0,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink max-buffers=1 drop=True"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

# load haarcascade untuk deteksi wajah dengan opencv
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# flip image
cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=2), cv2.CAP_GSTREAMER)

#inisialisasi model
from keras.models import model_from_json
model = model_from_json(open("model.json", "r").read())
model.load_weights('model_weights.h5')

#-----------------------------
emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')

def show_camera():
  if cap.isOpened():
    window_handle = cv2.namedWindow("CSI Camera", cv2.WINDOW_AUTOSIZE)
    # Window
    while cv2.getWindowProperty("CSI Camera", 0) >= 0:
      ret_val, img = cap.read() # capture frame video stream
      gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      faces = face_cascade.detectMultiScale(gray, 1.3, 5)
      for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
  
        detected_face = img[int(y):int(y+h), int(x):int(x+w)] 
        detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY) #ubah ke gray scale
        detected_face = cv2.resize(detected_face, (48, 48)) 
    
        img_pixels = image.img_to_array(detected_face)
        img_pixels = np.expand_dims(img_pixels, axis = 0)
  
        img_pixels /= 255 
    
        predictions = model.predict(img_pixels) 
    
        max_index = np.argmax(predictions[0])
    
        emotion = emotions[max_index]
    
        cv2.putText(img, emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        #mengirim hasil deteksi ke telegram bot     
        def handle(msg): 
          command = msg['text'] 
          print ('Got command: %s' % command) 
          # Membandingkan pesan yang masuk untuk mengirim balasan yg sesuai
          if command == '/help': 
            bot.sendMessage(chat_id,'ini adalah notifikasi dari monitoring emosi negatif pada anak autisme') 
          elif command == '/monitoring': 
            bot.sendMessage(chat_id, emotion) 
         #setiap kali pesan diterima, fungsi akan dipanggil
        MessageLoop(bot, handle).run_as_thread() 
        print ('Listening ...') 
        if emotion == 'angry':
          bot.sendMessage(chat_id, 'Awas tantrum, emosi anak sedang marah')
          sleep(2)
        elif emotion == 'sad':
          bot.sendMessage(chat_id, 'Awas tantrum, emosi anak sedang sedih')
          sleep(2)
        elif emotion == 'fear':
          bot.sendMessage(chat_id, 'Awas tantrum, emosi anak sedang takut')
          sleep(2)
        break
      cv2.imshow("CSI Camera", img) #show frame
      keyCode = cv2.waitKey(30) & 0xFF
      # tekan ESC key untuk stop
      if keyCode == 27:
        break
      
    cap.release()
    cv2.destroyAllWindows()
  else:
    print("Unable to open camera")
        
show_camera()




