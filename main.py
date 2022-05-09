import os
import sys

from PyQt5.QtGui import QMovie, QIcon
from PyQt5.QtCore import QUrl
from PyQt5.uic import loadUi
from PyQt5 import QtWidgets, QtGui, QtTest
from PyQt5.QtWidgets import QDialog, QApplication, QMainWindow
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent

import random


# Fo ablak
class WelcomeScreen(QMainWindow):
    def __init__(self):
        super(WelcomeScreen, self).__init__()
        # .ui file betoltese
        loadUi("Jatek.ui", self)
        # Egy kockas dobas dialog betoltese gomb megnyomasara
        self.OneDiceButton.clicked.connect(self.goto_one_play)
        # Meg nem letezo ket kockas dialog
        self.TwoDiceButton.clicked.connect(self.goto_two_play)
        # Exit gombra close_event fgv meghivasa
        self.ExitButton.clicked.connect(self.close_event)

        self.HomePage_Pixmap.setPixmap(QtGui.QPixmap("HomePage.png"))

    # Egy kockas dialog megnyitasa
    def goto_one_play(self):
        one_play = OneDicePlay()
        widget.addWidget(one_play)
        widget.setCurrentIndex(widget.currentIndex() + 1)

    # Ket kockas dobas dialog megnyitasa - nem letezo meg
    def goto_two_play(self):
        pass

    # Kilepes a programbol
    def close_event(self):
        sys.exit(0)


# Egy kockas dobas dialog letrehozasa
class OneDicePlay(QDialog):
    def __init__(self):
        super(OneDicePlay, self).__init__()
        # .ui betoltese
        loadUi("OneDice.ui", self)
        # Homepage gomb megnyomasara a kezdo mainwindow jon be
        self.HomePageButton.clicked.connect(self.goto_home)
        # Rolling gombra meghivodik a rolling fgv
        self.RollButton.clicked.connect(self.rolling)
        # Exit gombra close_event fgv meghivasa
        self.ExitButton.clicked.connect(self.close_event)

        # Hang pushbutton ki-be kapcsolhato legyen
        self.MuteButton.setCheckable(True)
        self.MuteButton.setIcon(QIcon("UNMUTE.png"))

        self.MuteButton.clicked.connect(self.volume_mute)

        self.BackgroundLabel.setPixmap(QtGui.QPixmap("Rolling_background.png"))

        self.movie = QMovie()

        self.audio = QMediaPlayer()

    # Kilepes a programbol
    def close_event(self):
        sys.exit(0)

    # Kezdo mainwindow betoltese (__ini__-ben definialt gomb nyomasara)
    def goto_home(self):
        # Fomenube visszatereskor minden eddigi valtoztatas (pl rolling gomb felirata) eredetire allitasa
        self.set_to_default()
        home = WelcomeScreen()
        widget.addWidget(home)
        widget.setCurrentIndex(widget.currentIndex() - 1)

    # Maganak a kockadobasnak a fgv-e
    def rolling(self):
        # Dobas elejen nem tudjuk mi lesz az eredmeny, de a label mar letezik
        self.ResultLabel.setText("")
        # Egy kocka dobasnal 1-6 kozotti egesz veletlenszam generalasa
        rolled = random.randint(1, 6)
        print("rolled", rolled) # FOR TESTING

        # A random szamtol fuggoen hivodik meg a kovetkezo fgv
        self.result_show(rolled)

    # Dobos video (vagy gif, meg elvalik) lejatszasa
    # number argument megfelel a dobott szamnak, azt kapja meg a self.result_show fgv-tol
    def rolling_video(self, number):
        # Szotar a megfelelo dobott szam -> video/ gif nevevel
        gif_dict = {1: "DICE_1.gif",
                    2: "DICE_2.gif",
                    3: "DICE_3.gif",
                    4: "DICE_4.gif",
                    5: "DICE_5.gif",
                    6: "DICE_6.gif"}
        # Egyelore uresen allo label widget eloterbe hozasa
        self.AnimationLabel.show()
        # A video lejatszasanak idejere eltuntetunk minden mast, gombokat, feliratokat
        # (kiveve a fooldal es exit gombokat)
        self.RollButton.hide()
        # Video betoltese - EGYELORE MEG CSAK EGY FELE
        self.movie = QMovie(gif_dict[number])
        self.AnimationLabel.setMovie(self.movie)

        # GIFhez tartozo audio fajl lejatszasa
        self.play_audio()
        # Video lejatszasa
        self.movie.start()

    # Mivel video helyett GIF formatumu a video, audio fajl kulon tartozik hozza, ami egyszerre indul vele
    def play_audio(self):
        full_filepath = os.path.join(os.getcwd(), "NEW_SOUND.wav")
        url = QUrl.fromLocalFile(full_filepath)

        content = QMediaContent(url)
        self.audio.setMedia(content)

        self.audio.play()

    # Ha le van nemitva az audio, visszakapcsolja; ha nincs, lenemitja azt
    def volume_mute(self):
        self.audio.setMuted(not self.audio.isMuted())

        # HA a gonb be van kapcsolva..
        if self.MuteButton.isChecked():
            # .. akkor nemit ikon legyen..
            self.MuteButton.setIcon(QIcon("MUTE.png"))

        # .. egyebkent..
        else:
            # .. hangszoro ikon
            self.MuteButton.setIcon(QIcon("UNMUTE.png"))

    # Ertekek alaphelyzetbe allitasa
    def set_to_default(self):
        # Elso dobas utan a gomb atvalt "ROLL AGAIN" feliratra. Fomenube kilepes utan, ha ujra dobalozni kezdunk, ismet
        # "ROLL THE DICE" felirat van a gombon
        self.RollButton.setText("Roll the dice!")
        self.RollButton.show()
        self.AnimationLabel.hide()
        # A hatter visszaallitasa
        self.BackgroundLabel.setPixmap(QtGui.QPixmap("Rolling_background.png"))
        # A dobott szamot sem tudjuk meg
        self.ResultLabel.setText("")
        self.audio.stop()

    def result_show(self, number):
        # Megfelelo video/ gif lejatszasa
        self.rolling_video(number)
        # Ne csinaljon semmit, amig a video/ gif vegig nem porgott (kb 4 mp) -< EZ BUGOS
        QtTest.QTest.qWait(4200)
        self.ResultLabel.setText(f"You rolled {number}")
        # Irja ki a dobott erteket, de ujra dobni csak fel mp mulva lehessen
        # (illetve az ujra dobos gomb megjelenese kessen) -> EZ BUGOS
        QtTest.QTest.qWait(500)
        self.RollButton.show()
        # Mivel mar dobtunk, a dobos gomb neve ne "dobj egyet" legyen, hanem "dobj ujra"
        self.RollButton.setText("Roll again!")


# main
app = QApplication(sys.argv)
welcome = WelcomeScreen()
widget = QtWidgets.QStackedWidget()
widget.addWidget(welcome)
widget.setFixedHeight(720)
widget.setFixedWidth(720)
widget.show()
try:
    sys.exit(app.exec_())
except:
    print("Exiting")
