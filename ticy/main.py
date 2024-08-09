import numpy as np

import kivy
from kivy.app import App
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.gridlayout import GridLayout

from model_script import *

model = Model()

kivy.require('2.3.0')

class MyApp(App):
    def build(self):
        self.ids={}
        self.titles = np.zeros((3,3))
        # Create a layout
        layout = GridLayout(cols=3, rows=3)

        for r in range(3):
            for c in range(3):
                button = Button(text='')
                button.r=r
                button.c=c
                button.bind(on_press=self.on_button_press)
                self.ids[f"button_{r}_{c}"]=button
                layout.add_widget(button)
        return layout

    def on_button_press(self, instance):
        instance.text = 'x'
        self.titles[instance.r,instance.c]=-1
        self.robot_play()
        print(self.titles)

    def robot_play(self):
        x_y=model.play(self.titles)
        self.titles[x_y[1], x_y[0]]=1
        self.ids[f"button_{x_y[1]}_{x_y[0]}"].text="o"

if __name__ == '__main__':
    MyApp().run()
