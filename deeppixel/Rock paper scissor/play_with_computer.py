from keras.models import load_model
from keras.models import model_from_json
from keras.optimizers import Adam
import keras
import cv2
import numpy as np
from random import choice

REV_CLASS_MAP = {
    0: "rock",
    1: "paper",
    2: "scissors",
    3: "none"
}


def mapper(val):
    return REV_CLASS_MAP[val]


def calculate_winner(move1, move2):
    if move1 == move2:
        return "Tie"

    if move1 == "rock":
        if move2 == "scissors":
            return "User"
        if move2 == "paper":
            return "Computer"

    if move1 == "paper":
        if move2 == "rock":
            return "User"
        if move2 == "scissors":
            return "Computer"

    if move1 == "scissors":
        if move2 == "paper":
            return "User"
        if move2 == "rock":
            return "Computer"

json_file = open('model.json' , 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model = load_model("rock_paper_scissor_new.h5")

model.compile(
    optimizer=Adam(lr=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
cap = cv2.VideoCapture(0)

prev_move = None

ctr = 0
while True:

    
   
    ret, frame = cap.read()
    if not ret:
        continue

    #rectangle for making a rectangle in border
    cv2.line(frame, (0,0), (0,480), (255 ,255,255), 50) 
    cv2.line(frame, (0,0), (640,0), (255 ,255,255), 224) 
    cv2.line(frame, (0,480), (640,480), (255 ,255,255), 150) 
    cv2.line(frame, (640,0), (640,480), (255 ,255,255), 550) 


    
    # rectangle for user to play
    cv2.rectangle(frame, (20 ,  120), (300, 400), (255, 255, 255), 17)
    
    # rectangle for computer to play
    # cv2.rectangle(frame, (350, 120), (630, 400), (0, 0, 255), 10)

    cv2.rectangle(frame, (320, 120), (600, 400), (255,255, 255), 20)
    


    # extract the region of image within the user rectangle
    roi = frame[120:400, 20:300]
    img = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (227, 227))

    # predict the move made
    pred = model.predict(np.array([img]))
    move_code = np.argmax(pred[0])
    # move_code =0
    # if ctr%10==0:     
    #     move_code =  np.random.randint(3)
    # print(pred)
    user_move_name = mapper(move_code)
    # ctr +=1

    #predict the winner (human vs computer)
    if prev_move != user_move_name:
        if user_move_name != "none":
            computer_move_name = choice(['rock', 'paper', 'scissors'])
            winner = calculate_winner(user_move_name, computer_move_name)
        else:
            computer_move_name = "none"
            winner = "Waiting..."
    prev_move = user_move_name

    #display the information
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, "Your Move : " + user_move_name,
                (50, 50), font, 0.7, (255, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "Computer's Move: " + computer_move_name,
                (330, 50), font, 0.7, (255, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "Winner: " + winner,
                (150, 450 ), font,1.5, (0, 0, 255), 4, cv2.LINE_AA)
    
    if computer_move_name != "none":
        icon = cv2.imread(
            "images/{}.png".format(computer_move_name))
        icon = cv2.resize(icon, (280, 280))
        frame[120:400,320:600] = icon
        frame[120:400,20:300] = roi
        print(pred)

    cv2.namedWindow("Rock Paper Scissors",0) 
    cv2.resizeWindow('Rock Paper Scissors', 1000,1000)
    cv2.imshow("Rock Paper Scissors", frame)
    
   



    k = cv2.waitKey(10)
    if k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
