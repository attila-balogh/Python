from tkinter import *
from PIL import ImageTk, Image
from random import *

root = Tk()
root.title("Amoba")
root.iconbitmap("amoeba.ico")

# DEFAULT FONT
FONT = ('Ariel', 14)

# X=1, O=0
starter = True
winner_player_X = False
winner_player_O = False
step = 0
win = False

# LOGIC TABLE FOR X AND O
logic_table_X = [[0 for i in range(15)] for j in range(15)]
logic_table_O = [[0 for i in range(15)] for j in range(15)]

# CREATING AND DRAWING WAR TABLE |15 × 15|
buttons = [[0 for i in range(15)] for j in range(15)]

for i in range(15):
    column_label = Label(text=(chr(i + 97).upper()), font=FONT)
    column_label.grid(row=4,column=2+i)

for i in range(15):
    row_label = Label(text=1+i, font=FONT)
    row_label.grid(row=5+i,column=1)

for row in range(5,20):
    for column in range(2, 17):
        button = Button(root, width=5, height=2, font=('bold', 12),
                        command=lambda row=row, column=column: button_clicked(row-5, column-2))
        button.grid(row=row, column=column)
        buttons[row-5][column-2] = button


# IF CLICKED, MAKE THAT BUTTON INACTIVE, AND GIVE IT COLOR AND SIGNAL X OR O
def button_clicked(row, column):
    global step
    global winner_player_O
    global winner_player_X
    global starter
    global buttons
    step += 1
    if logic_table_X[row][column] == 0 and logic_table_O[row][column] == 0:
        if starter:
            buttons[row][column].config(state="disabled", text='X', bg='red', fg='red')
            logic_table_X[row][column] = 1
        elif not starter:
            buttons[row][column].config(state="disabled", text='O', bg="blue", fg='blue')
            logic_table_O[row][column] = 1
    # CHANGE PLAYER
    starter = not starter

    # TEST IF O WINS
    if test_O_win():
        for row in range(15):
            for column in range(15):
                buttons[row][column].config(state="disable")
        winner_player_O = True
        winner_label.config(text="The Winner is")
        winner_label2.config(text="Player O", fg='blue')

    # TEST IF X WINS
    if test_X_win():
        for row in range(15):
            for column in range(15):
                buttons[row][column].config(state="disable")
        winner_player_X = True
        winner_label.config(text="The Winner is")
        winner_label2.config(text="Player X", fg='red')

    # TEST IF NO MORE STEPS
    if step == 15*15 and not winner_player_X and not winner_player_O:
        winner_label.config(text="No more steps!", fg='black')
        winner_label2.config(text="It's a TIE", fg='black')


def test_X_win():
    global win
    win = False
    for i in range(15):                 # HORIZONTAL
        for j in range(12):
            win = logic_table_X[i][j] * logic_table_X[i][j+1] * logic_table_X[i][j+2] * logic_table_X[i][j+3]
            if win:
                buttons[i][j].config(bg="sienna1")
                buttons[i][j+1].config(bg="sienna1")
                buttons[i][j+2].config(bg="sienna1")
                buttons[i][j+3].config(bg="sienna1")
                return True
    for i in range(12):                 # VERTICAL
        for j in range(15):
            win = logic_table_X[i][j] * logic_table_X[i+1][j] * logic_table_X[i+2][j] * logic_table_X[i+3][j]
            if win:
                buttons[i][j].config(bg="sienna1")
                buttons[i + 1][j].config(bg="sienna1")
                buttons[i + 2][j].config(bg="sienna1")
                buttons[i + 3][j].config(bg="sienna1")
                return True
    for i in range(12):                 # DIAGONAL1
        for j in range(12):
            win = logic_table_X[i][j] * logic_table_X[i+1][j+1] * logic_table_X[i+2][j+2] * logic_table_X[i+3][j+3]
            if win:
                buttons[i][j].config(bg="sienna1")
                buttons[i + 1][j + 1].config(bg="sienna1")
                buttons[i + 2][j + 2].config(bg="sienna1")
                buttons[i + 3][j + 3].config(bg="sienna1")
                return True
    for i in range(12):                 # DIAGONAL2
        for j in range(3, 15):
            win = logic_table_X[i][j] * logic_table_X[i+1][j-1] * logic_table_X[i+2][j-2] * logic_table_X[i+3][j-3]
            if win:
                buttons[i][j].config(bg="sienna1")
                buttons[i + 1][j - 1].config(bg="sienna1")
                buttons[i + 2][j - 2].config(bg="sienna1")
                buttons[i + 3][j - 3].config(bg="sienna1")
                return True
    return win


def test_O_win():
    global win
    win = False
    for i in range(15):                 # HORIZONTAL
        for j in range(12):
            win = logic_table_O[i][j] * logic_table_O[i][j+1] * logic_table_O[i][j+2] * logic_table_O[i][j+3]
            if win:
                buttons[i][j].config(bg="SkyBlue1")
                buttons[i][j + 1].config(bg="SkyBlue1")
                buttons[i][j + 2].config(bg="SkyBlue1")
                buttons[i][j + 3].config(bg="SkyBlue1")
                return True
    for i in range(12):                 # VERTICAL
        for j in range(15):
            win = logic_table_O[i][j] * logic_table_O[i+1][j] * logic_table_O[i+2][j] * logic_table_O[i+3][j]
            if win:
                buttons[i][j].config(bg="SkyBlue1")
                buttons[i + 1][j].config(bg="SkyBlue1")
                buttons[i + 2][j].config(bg="SkyBlue1")
                buttons[i + 3][j].config(bg="SkyBlue1")
                return True
    for i in range(12):                 # DIAGONAL1
        for j in range(12):
            win = logic_table_O[i][j] * logic_table_O[i+1][j+1] * logic_table_O[i+2][j+2] * logic_table_O[i+3][j+3]
            if win:
                buttons[i][j].config(bg="SkyBlue1")
                buttons[i + 1][j + 1].config(bg="SkyBlue1")
                buttons[i + 2][j + 2].config(bg="SkyBlue1")
                buttons[i + 3][j + 3].config(bg="SkyBlue1")
                return True
    for i in range(12):                 # DIAGONAL2
        for j in range(4, 15):
            win = logic_table_O[i][j] * logic_table_O[i+1][j-1] * logic_table_O[i+2][j-2] * logic_table_O[i+3][j-3]
            if win:
                buttons[i][j].config(bg="SkyBlue1")
                buttons[i + 1][j - 1].config(bg="SkyBlue1")
                buttons[i + 2][j - 2].config(bg="SkyBlue1")
                buttons[i + 3][j - 3].config(bg="SkyBlue1")
                return True
    return win


def reset():
    global step
    global winner_player_O
    global winner_player_X
    global starter
    global buttons
    global win
    global logic_table_X
    global logic_table_O

    win = False
    winner_player_X = False
    winner_player_O = False
    step = 0
    starter = 1

    # RESET LOGIC TABLE FOR X AND O
    logic_table_X = [[0 for i in range(15)] for j in range(15)]
    logic_table_O = [[0 for i in range(15)] for j in range(15)]

    # RESET WINNING TEXT
    winner_label.config(text=" ")
    winner_label2.config(text=" ")

    # RESET BUTTONS
    for row in range(5, 20):
        for column in range(2, 17):
            buttons[row-5][column-2].config(state="active", text="", bg='#F0F0F0')


# EXIT BUTTON
button_quit = Button(root, text="Exit", font='4', width=15, command=root.quit)
button_quit.grid(row=22, column=3, columnspan=3)

# NEW GAME BUTTON
button_new_game = Button(root, font='4', text="New Game", width=15, command=lambda: reset())
button_new_game.grid(row=22, column=13, columnspan=3)

# MAKE BORDER
empty_label = Label(root, text=" ", height=2)
empty_label.grid(row=21, column=0, columnspan=19)
# empty_label = Label(root, text=" ", height=2)
# empty_label.grid(row=1, column=0, columnspan=19)
empty_label = Label(root, text=" ", width=5)
empty_label.grid(row=1, column=0, rowspan=22)
empty_label = Label(root, text=" ", width=10)
empty_label.grid(row=1, column=18, rowspan=22)
empty_label = Label(root, text=" ", height=2)
empty_label.grid(row=23, column=0, columnspan=19)

# TITLE
title_label = Label(root, text = "AMOEBA", font=('Ariel', 24))
title_label.grid(row=0, column=0, columnspan=19)

# WINNING LABEL
winner_label = Label(root, text = " ", font=('Ariel', 24))
winner_label.grid(row=2, column=0, columnspan=19)
winner_label2 = Label(root, text = " ", font=('Ariel',36, 'bold'))
winner_label2.grid(row=3, column=0, columnspan=19)


root.mainloop()