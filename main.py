import json
import argparse
import utils.train as train
import utils.predict as predict
import utils.data as data

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config',
                        help="config model and dataset path",
                        default="config.json")
    parser.add_argument('-o', '--onnx_path', 
                        help='ONNX model path.',
                        default="best_model.onnx") 
    
    parser.add_argument('-p', '--pt_path',
                        help='pt model path',
                        default='best_model.pt')
    parser.add_argument('-d', '--data',
                        help='load csv file for train or inference',
                        default='')
    parser.add_argument('-s', '--save_dir',
                        help="save model path in train mode",
                        default='')
    
    parser.add_argument('-e',"--epochs",
                        help="set epochs for train mode",
                        default='1')
    parser.add_argument("--patience",
                        help="set patience for early stopping in train mode",
                        default='1')
    parser.add_argument('--save_onnx',
                        help="save ONNX file in training mode",
                        action="store_true")
    parser.add_argument("--train",
                        action="store_true")
    parser.add_argument("--onnx_predict",
                        action="store_true")                        
    parser.add_argument("--pt_predict",
                        action="store_true")
    
    
    args = parser.parse_args()

    #load config.json file
    with open(args.config, 'r') as f:
        conf = json.load(f)

    #load dataset
    if args.data == "":
        df = data.ml100k_dataset()
    else:
        df = data.load_csv_dataset(args.data)

    #train model
    if args.train:
        train.train_model(conf,
                    epochs=int(args.epochs),
                    df=df,
                    save_dir=args.save_dir,
                    model_name=args.pt_path,
                    patience=int(args.patience))
        
    #load and ranking model from ONNX file
    onnx_model_path = args.save_dir + args.pt_path.split(".")[0] + ".onnx"
    if args.onnx_predict:
        ort_session = predict.init_onnx_model(onnx_model_path)
        out = predict.predict_from_onnx(ort_session)
        print(out)

    #load and ranking model from pt file
    if args.pt_predict:
        output = predict.predict_from_pt(args.pt_path,conf)
        print(output)