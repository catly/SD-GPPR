'''
Concrete SettingModule class for a specific experimental SettingModule
'''
import torch.cuda
# Copyright (c) 2017 Jiawei Zhang <jwzhanggy@gmail.com>
# License: TBD

from mycode.base_class.setting import setting

device = "cuda:0" if torch.cuda.is_available() else "cpu"
class Settings(setting):
    fold = None

    def load_run_save_evaluate(self):
        
        # load dataset
        loaded_data = self.dataset.load()

        # run learning methods
        self.method.data = loaded_data
        # self.method.to(device)
        learned_result = self.method.run()

        # save raw ResultModule
        self.result.data = learned_result
        self.result.save()

        # evaluate learning results
        if self.evaluate is not None:
            self.evaluate.data = learned_result
            self.evaluate.evaluate()

        return None

        