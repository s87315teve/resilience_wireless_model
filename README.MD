無人機空對地訊號強度預測模型
===

## **Model architecture**
![alt text](model_architecture.png)


## Usage
**Train**
```bash
python train.py --model_name GeoSigNet --dataset 3GPP_scenario_100k.csv --tag 100k_with_all_sensor --batch_size 512 --add_columns Distance,Angle,LOS_Flag,GPS --warmup_factor 10 --warmup_epochs 10
```


**Test**
```bash 
python batch_test.py --model_name GeoSigNet --model_folder 100k_with_all_sensor_epoch100 --dataset 3GPP_scenario_100k.csv --tag 30m_with_all_sensor  --add_columns Distance,Angle,LOS_Flag,GPS
```

