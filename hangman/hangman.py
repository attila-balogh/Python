from tkinter import *
from PIL import ImageTk, Image
from random import *

root = Tk()
root.title("Hangman")
root.iconbitmap("icon.ico")


default = ['dog', 'cat', 'lamb', 'football', 'pitch', 'game', 'organ', 'guitar', 'table', 'tennis', 'desk', 'home',
            'science', 'chair', 'lion', 'balcony', 'umbrella', 'glass', 'water', 'yellow', 'book', 'television', 'tiger',
            'mouse', 'computer']


def setImage(error):
    if error == 0:
        return "1.png"
    if error == 1:
        return "2.png"
    if error == 2:
        return "3.png"
    if error == 3:
        return "4.png"
    if error == 4:
        return "5.png"
    if error == 5:
        return "6.png"
    if error == 6:
        return "7.png"
    if error == 7:
        return "8.png"


word = (default[randint(0, len(default) - 1)])

playa = []
for i in word:
    playa.append(i.upper())

guess_word = []
for i in range(len(word)):
    guess_word.append(' _ ')


buttons = []
guesses = []
error = 0
img = []
solved = ''
for i in word:
    solved += ' _ '

result_label = Label(text=" ", font=('Ariel', 16))
result_label.grid(row=3, columnspan=9)

solved_label = Label(text=solved, font=('Ariel', 20))
solved_label.grid(row=0, columnspan=9)


my_img = ImageTk.PhotoImage(Image.open(setImage(error)))
hangmanPic = Label(image=my_img)
hangmanPic.grid(row=1, column=0, columnspan=9)

img.append(hangmanPic)

def button_clicked(index, n):
    global error
    global img
    global guesses
    global word
    guesses.append(n)
    if n in word.upper():
        buttons[index].config(state="disabled", fg='white', bg='green')
        solved_disp = ''
        for k in range(len(word)):
            if word.upper()[k] == n:
                guess_word[k] = n
            solved_disp += guess_word[k]
        solved_label.config(text=solved_disp)
    else:
        buttons[index].config(state="disabled", fg='white', bg='red')
        error += 1
        img.append(error)
        img[error] = ImageTk.PhotoImage(Image.open(setImage(error)))
        hangmanPic.config(image=img[error])
    if error == 7:
        for p in range(len(letters)):
            buttons[p].config(state="disabled")
        solved_label.config(text=word.upper())
        result_label.config(text="You lost!")
    elif guess_word == playa:
        for p in range(len(letters)):
            buttons[p].config(state="disabled")
        img.append(error+1)
        img[error+1] = ImageTk.PhotoImage(Image.open("win.png"))
        hangmanPic.configure(image=img[error+1])
        result_label.config(text="You won!")


letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S',
           'T', 'U', 'V','W', 'X', 'Y', 'Z', '-']


for index in range(27):
    n = letters[index]
    button = Button(root, bg="#F0F0F0", fg='black', font='5', text=n, width=5, command=lambda index=index, n=n: button_clicked(index, n))
    button.grid(padx=2, pady=2, row=int(index/9+4), column=int(index%9))
    buttons.append(button)


button_quit = Button(root, text="Exit", font='4', width=20, command=root.quit)
button_quit.grid(row=8, column=0, columnspan=3)

empty_label = Label(root, font='4', text=" ")
empty_label.grid(row=7, column=0, columnspan=9)

button_new_game = Button(root, font='4', text="New Game", width=20, command=lambda: reset())
button_new_game.grid(row=8, column=6, columnspan=3)


def reset():
    global error
    global img
    global guesses
    global word
    global buttons
    global solved
    global guess_word
    global playa
    global my_img
    global hangmanPic
    global solved_disp
    error = 0

    result_label.config(text=" ")

    word = (default[randint(0, len(default) - 1)])

    playa = []
    for i in word:
        playa.append(i.upper())

    guess_word = []

    for i in range(len(word)):
        guess_word.append(' _ ')

    guesses = []

    solved = ''
    for i in word:
        solved += ' _ '

    solved_disp = ''

    for k in range(len(word)):
        if word.upper()[k] == n:
            guess_word[k] = n
        solved_disp += guess_word[k]
    solved_label.config(text=solved_disp)

    img.clear()
    my_img = ImageTk.PhotoImage(Image.open(setImage(error)))
    hangmanPic = Label(image=my_img)
    hangmanPic.grid(row=1, column=0, columnspan=9)

    img.append(hangmanPic)

    for index in range(27):
        buttons[index].config(state="active", bg='#F0F0F0', fg='black')


root.mainloop()