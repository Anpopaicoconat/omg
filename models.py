import torch
import pytorch_lightning as pl
import torchmetrics

class tfidf_model(pl.LightningModule):
    def __init__(self, n_inp, n_clasess, DR=0.4):
        super().__init__()
        self.D1 = torch.nn.Linear(n_inp, 64, dtype=float)
        self.D2 = torch.nn.Linear(64, 64, dtype=float)
        self.D3 = torch.nn.Linear(64, n_clasess, dtype=float)
        self.drop = torch.nn.Dropout(DR)

        #metrics
        self.f1_train = torchmetrics.F1Score()
        self.f1_val = torchmetrics.F1Score()
        self.acc_train = torchmetrics.Accuracy()
        self.acc_val = torchmetrics.Accuracy()

        #loss
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.D1(x)
        x = torch.nn.functional.relu(x)
        x = self.drop(x)
        x = self.D2(x)
        x = torch.nn.functional.relu(x)
        x = self.drop(x)
        x = self.D3(x)
        #x = torch.nn.functional.softmax(x, dim=0)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-2)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        y_pred = self(x)    

        loss = self.loss(y_pred, y)
        self.log('train_loss', loss)
        self.f1_train(y_pred, y)
        self.acc_train(y_pred, y)
        return loss

    def training_epoch_end(self, outs):
        f1 = self.f1_train.compute()
        acc = self.acc_train.compute()
        self.log('train_f1_epoch', f1)
        self.log('train_acc_epoch', acc)

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        y_pred = self(x)


        loss = self.loss(y_pred, y)
        self.f1_val(y_pred, y)
        self.acc_val(y_pred, y)
        self.log('val_loss', loss)

    def validation_epoch_end(self, outs):
        f1 = self.f1_val.compute()
        acc = self.acc_val.compute()
        self.log('val_f1_epoch', f1)
        self.log('val_acc_epoch', acc)

class lstm_model(pl.LightningModule):
    def __init__(self, n_inp, n_clasess, DR=0.4):
        super().__init__()
        self.embd = torch.nn.Embedding(100, 7, padding_idx=0)
        self.D1 = torch.nn.LSTM(64, 64, dtype=float, batch_first=True)
        self.D2 = torch.nn.LSTM(64, 64, dtype=float, batch_first=True)
        self.D3 = torch.nn.Linear(64, n_clasess, dtype=float)
        self.drop = torch.nn.Dropout(DR)

        #metrics
        self.f1_train = torchmetrics.F1Score()
        self.f1_val = torchmetrics.F1Score()
        self.acc_train = torchmetrics.Accuracy()
        self.acc_val = torchmetrics.Accuracy()

        #loss
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.embd(x)
        x = self.D1(x)
        x = torch.nn.functional.relu(x)
        x = self.drop(x)
        x = self.D2(x)
        x = torch.nn.functional.relu(x)
        x = self.drop(x)
        x = self.D3(x[:, :, -1])
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-2)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        y_pred = self(x)    

        loss = self.loss(y_pred, y)
        self.log('train_loss', loss)
        self.f1_train(y_pred, y)
        self.acc_train(y_pred, y)
        return loss

    def training_epoch_end(self, outs):
        f1 = self.f1_train.compute()
        acc = self.acc_train.compute()
        self.log('train_f1_epoch', f1)
        self.log('train_acc_epoch', acc)

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        y_pred = self(x)


        loss = self.loss(y_pred, y)
        self.f1_val(y_pred, y)
        self.acc_val(y_pred, y)
        self.log('val_loss', loss)

    def validation_epoch_end(self, outs):
        f1 = self.f1_val.compute()
        acc = self.acc_val.compute()
        self.log('val_f1_epoch', f1)
        self.log('val_acc_epoch', acc)


class pl_cnn2d(pl.LightningModule):
    def __init__(self, y_kernel, n_mels, n_input=1, n_output=2, stride=(128, 1), n_channel=32, DR=0.2):
        super().__init__()
        self.conv1 =torch.nn.Conv2d(n_input, n_channel, kernel_size=(n_mels, y_kernel), stride=stride, padding=(0, y_kernel//2))
        self.bn1 =torch.nn.BatchNorm2d(n_channel)
        self.pool1 =torch.nn.MaxPool2d((1, 4))
        self.drop1 =torch.nn.Dropout(DR)

        self.conv2 =torch.nn.Conv2d(n_channel, n_channel, kernel_size=(1, 3))
        self.bn2 =torch.nn.BatchNorm2d(n_channel)
        self.pool2 =torch.nn.MaxPool2d((1, 4))
        self.drop2 =torch.nn.Dropout(DR)

        self.conv3 =torch.nn.Conv2d(n_channel, 2 * n_channel, kernel_size=(1, 3))
        self.bn3 =torch.nn.BatchNorm2d(2 * n_channel)
        self.pool3 =torch.nn.MaxPool2d((1, 4))
        self.drop3 =torch.nn.Dropout(DR)

        self.drop4 =torch.nn.Dropout(DR)
        self.fc4 =torch.nn.Linear(512, n_output)

        #metrics
        self.f1_train = torchmetrics.F1Score()
        self.f1_val = torchmetrics.F1Score()
        self.acc_train = torchmetrics.Accuracy()
        self.acc_val = torchmetrics.Accuracy()

        #loss
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.conv1(x)
        x = torch.nn.functional.relu(self.bn1(x))
        x = self.pool1(x)
        x = self.drop1(x)

        x = self.conv2(x)
        x = torch.nn.functional.relu(self.bn2(x))
        x = self.pool2(x)
        x = self.drop2(x)

        x = self.conv3(x)
        x = torch.nn.functional.relu(self.bn3(x))
        x = self.pool3(x)
        x = self.drop3(x)
        
        x = torch.flatten(x, 1)
        x = self.drop4(x)
        x = self.fc4(x)
        return torch.nn.functional.log_softmax(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-2)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        y_pred = self(x)    

        loss = self.loss(y_pred, y)
        self.log('train_loss', loss)
        self.f1_train(y_pred, y)
        self.acc_train(y_pred, y)
        return loss

    def training_epoch_end(self, outs):
        f1 = self.f1_train.compute()
        acc = self.acc_train.compute()
        self.log('train_f1_epoch', f1)
        self.log('train_acc_epoch', acc)
        print('f1', f1, 'acc', acc)

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        y_pred = self(x)


        loss = self.loss(y_pred, y)
        self.f1_val(y_pred, y)
        self.acc_val(y_pred, y)
        self.log('val_loss', loss)

    def validation_epoch_end(self, outs):
        f1 = self.f1_val.compute()
        acc = self.acc_val.compute()
        self.log('val_f1_epoch', f1)
        self.log('val_acc_epoch', acc)
        print('f1', f1, 'acc', acc)
        
class pl_efnet(pl.LightningModule):
    def __init__(self, le, extractor, main_modul, n_input=1, n_output=41):
        super().__init__()
        self.le = le
        self.extractor = extractor

        
        self.main_modul = main_modul

        self.out_activation = torch.nn.LogSoftmax()

        #loass & metrics
        self.f1_train = torchmetrics.F1Score(num_classes=n_output)
        self.f1_val = torchmetrics.F1Score(num_classes=n_output)
        self.acc_train = torchmetrics.Accuracy(num_classes=n_output)
        self.acc_val = torchmetrics.Accuracy(num_classes=n_output)
        self.loss = torch.nn.CrossEntropyLoss()

        # self.automatic_optimization = False

    def forward(self, x):
        x = self.extractor(x)
        x = self.main_modul(x)
        x = self.out_activation(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=0.5,  total_iters=50)
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        y_pred = self(x)
        loss = self.loss(y_pred, y)
        self.log('train_loss', loss)
        self.f1_train(y_pred, y)
        self.acc_train(y_pred, y)
        return loss

    def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
        scheduler.step()

    def training_epoch_end(self, outs):
        sch = self.lr_schedulers()
        sch.step()

        f1 = self.f1_train.compute()
        acc = self.acc_train.compute()
        self.log('train_f1_epoch', f1)
        self.log('train_acc_epoch', acc)
        print('train_f1', f1, 'train_acc', acc)

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        y_pred = self(x)
        loss = self.loss(y_pred, y)
        y_pred = torch.argmax(y_pred, dim=1)
        self.f1_val(y_pred, y)
        self.acc_val(y_pred, y)
        self.log('val_loss', loss)

    def validation_epoch_end(self, outs):
        f1 = self.f1_val.compute()
        acc = self.acc_val.compute()
        self.log('val_f1_epoch', f1)
        self.log('val_acc_epoch', acc)
        print('val_f1', f1, 'val_acc', acc)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        y_pred = self(x)
        y_pred = torch.argmax(y_pred, dim=1)
        y_pred = self.le.inverse_transform(y_pred.cpu().detach().numpy())
        return y_pred
# it is main_modul    
#efnet7 = torchvision.models.efficientnet_b7(pretrained=True)
#efnet7.features[0][0] = torch.nn.Conv2d(in_channels, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False) #64
#efnet7.classifier[1] = torch.nn.Linear(in_features=2560, out_features=n_classes, bias=True) #2560
