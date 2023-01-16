from ultralytics import YOLO

model = YOLO()
model.resume(task="detect") # resume last detection training
# model.resume(model="last.pt") # resume from a given model/run

# resume()