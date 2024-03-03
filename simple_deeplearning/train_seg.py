import torch
from torch.utils.data import DataLoader
from datasets.seg_set import SegSet
from models.tiny_model import TinyModel

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

if __name__ == '__main__':
    print('Train Seg')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using {} device'.format(device))

    model = TinyModel().to(device)

    learning_rate = 1e-3
    batch_size = 5
    epochs = 2000
    step_size = 200

    training_data = SegSet('train_samples')
    test_data = SegSet('test_samples')

    training_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma = 0.9)

    seg_loss_fn = torch.nn.CrossEntropyLoss()
    
    def loop(dataloader, model, mode='train'):
        num_batches = len(dataloader)

        if mode == 'train':
            model.train()
            torch.set_grad_enabled(True)
        elif mode == 'test':
            torch.set_grad_enabled(False)
        
        mean_loss = 0
        for batch_idx, (x_img, y_seg) in enumerate(dataloader):
            x_img = x_img.to(device)
            y_seg = y_seg.to(device)

            pred_seg = model(x_img)

            loss = seg_loss_fn(pred_seg, y_seg)

            if mode == 'train':
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            mean_loss += loss.item()

        mean_loss /= num_batches
        
        return mean_loss
    
    min_train_loss = 1e5
    min_test_loss = 1e5

    for t in range(epochs):
        train_loss = loop(training_dataloader, model, 'train')
        test_loss = loop(test_dataloader, model, 'test')

        writer.add_scalar('loss/train', train_loss, t)
        writer.add_scalar('loss/test', test_loss, t)

        if (lr_scheduler.get_last_lr()[0] > 1e-7):
            lr_scheduler.step()

        if t % 10 == 0:
            print('epoch: {} lr: {}'.format(t, lr_scheduler.get_last_lr()))
            print('train loss: {}'.format(train_loss))
            print('test loss: {}'.format(test_loss))

        if train_loss < min_train_loss:
            min_train_loss = train_loss
            torch.save(model.state_dict(), 'weights/weights_min_train_loss.pth')

        if test_loss < min_test_loss:
            min_test_loss = test_loss
            print('min loss: {}'.format(min_test_loss))
            torch.save(model.state_dict(), 'weights/weights_min_test_loss.pth')

    writer.close()


