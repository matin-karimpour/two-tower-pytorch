import onnxruntime
import torch
import json
import model.TwoTower as TwoTower

def init_onnx_model(model_path:str):
    ort_session = onnxruntime.InferenceSession(model_path)
    return ort_session

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def predict_from_onnx(ort_session, 
                      input=[
                        torch.tensor([[193,  29]]),
                        torch.tensor([[147,  1]])
                        ]
    ):
    ort_inputs = {"input_user": to_numpy(input[0]),
                  "input_item": to_numpy(input[1])}
    ort_outs = ort_session.run(None, ort_inputs)
    return ort_outs

def predict_from_pt(path,
                    conf,
                    input=[
                        torch.tensor([[193,  29]]),
                        torch.tensor([[147,  1]])
                        ]):
    model = TwoTower.TwoTower(conf)
    model.load_state_dict(torch.load(path))
    model.eval()
    output = model(input)
    return output

if __name__ == "__main__":
    # [tensor([[193,  29]]), tensor([[147,   1]])] 

    input = [
        torch.tensor([[193,  29]]),
        torch.tensor([[147,  1]])
    ]
    
    ort_session = init_onnx_model("best_twotower.onnx")
    out = predict_from_onnx(ort_session, input)
    print(out)


    