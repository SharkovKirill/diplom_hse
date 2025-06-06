import numpy as np
import torch
import pytorch_lightning as pl
import torch.nn as nn

import clip


img_path = "bad_est.jpg"  # 5.4315
img_path = "good_est.jpg"  # 6.0443


class MLP(pl.LightningModule):
    def __init__(self, input_size, xcol="emb", ycol="avg_rating"):
        super().__init__()
        self.input_size = input_size
        self.xcol = xcol
        self.ycol = ycol
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            # nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            # nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            # nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            # nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.layers(x)

    def training_step(self, batch, batch_idx):
        x = batch[self.xcol]
        y = batch[self.ycol].reshape(-1, 1)
        x_hat = self.layers(x)
        loss = F.mse_loss(x_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch[self.xcol]
        y = batch[self.ycol].reshape(-1, 1)
        x_hat = self.layers(x)
        loss = F.mse_loss(x_hat, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


def normalized(a, axis=-1, order=2):
    import numpy as np

    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


def get_mlp_model():
    model = MLP(768)  # CLIP embedding dim is 768 for CLIP ViT L 14
    try:
        s = torch.load("./ava_aesthetic_predictor/sac+logos+ava1-l14-linearMSE.pth")
    except Exception:
        s = torch.load("sac+logos+ava1-l14-linearMSE.pth")

    model.load_state_dict(s)

    model.to("cuda")
    model.eval()
    return model


def make_prediction_pretrained_ava(pil_image):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model2, preprocess = clip.load("ViT-L/14", device=device)

    image = preprocess(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = model2.encode_image(image)

    im_emb_arr = normalized(image_features.cpu().detach().numpy())
    model = get_mlp_model()
    prediction = model(
        torch.from_numpy(im_emb_arr).to(device).type(torch.cuda.FloatTensor)
    )

    print(f"Aesthetic score predicted by the model: {prediction}")
    return round(prediction.cpu().item(), 2)


# prediction = make_prediction_pretrained_ava(img_path)
# print(prediction)
