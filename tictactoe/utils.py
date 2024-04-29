class counter():
    def __init__(self, max=0):
        self.counter = 0
        self.max = max
        self.is_max = False
        if max != 0: self.is_max = True
        print('reset\n')
    
    def start(self):
        print('timer started\n')
        self.start_time=time.time()
    
    def stop(self):
        print(f'time: {round(time.time()-self.start_time)} s')

    def count(self):
        sys.stdout.write('\033[1A')
        sys.stdout.write('\033[1G\033[K')
        self.counter += 1
        if self.is_max:
            text = f'{self.counter}/{self.max}'
        else:
            text = str(self.counter)
        print(text)

    def reset(self, *args, **kwargs):
        if args:
            self.__init__(args[0])
        else:
            self.__init__(0)