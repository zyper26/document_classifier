
import torchvision.transforms as transforms

my_transforms = transforms.Compose(
    [
        transforms.Resize((64, 64)),  
        transforms.RandomRotation(
            degrees=45
        ),  
        transforms.RandomHorizontalFlip(
            p=0.5
        ), 
        transforms.RandomVerticalFlip(
            p=0.05
        ), 
        transforms.RandomGrayscale(p=0.2), 
        transforms.ToTensor()
    ]
)