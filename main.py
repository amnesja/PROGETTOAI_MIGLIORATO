import argparse
from scripts import train
from scripts import evaluate
from scripts import predict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "eval", "predict", "resume"], required=True)
    args = parser.parse_args()

    if args.mode == "train":
        train()

    elif args.mode == "eval":
        evaluate()

    elif args.mode == "predict":
        image_path = input("Inserisci il percorso dell'immagine per la predizione: ")
        predict(image_path)
    
    elif args.mode == "resume":
        train(resume=True)

if __name__ == "__main__":
    main()