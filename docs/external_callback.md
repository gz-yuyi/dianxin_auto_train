
# 专用模型训练通知

## POST 专用模型训练-每轮训练完成结果通知接口

POST /api/model/train/notify/result

> Body 请求参数

```json
{
  "trainTaskId": 0,
  "epoch": 0,
  "accuracy": 0,
  "loss": 0,
  "valAccuracy": 0,
  "valLoss": 0,
  "f1Score": 0
}
```

### 请求参数

|名称|位置|类型|必选|说明|
|---|---|---|---|---|
|body|body|[SpecialModelTrainResultNotifyReq](#schemaspecialmodeltrainresultnotifyreq)| 否 |none|

> 返回示例

> 200 Response

```json
{
  "success": false,
  "code": 0,
  "msg": "",
  "data": ""
}
```

### 返回结果

|状态码|状态码含义|说明|数据模型|
|---|---|---|---|
|200|[OK](https://tools.ietf.org/html/rfc7231#section-6.3.1)|none|[SingleResponseString](#schemasingleresponsestring)|


## POST 专用模型训练-训练状态变更通知

POST /api/model/train/notify/status

> Body 请求参数

```json
{
  "trainTaskId": "string",
  "statusType": 0,
  "failureMessage": "string"
}
```

### 请求参数

|名称|位置|类型|必选|说明|
|---|---|---|---|---|
|body|body|[SpecialModelTrainStatusChangeNotifyReq](#schemaspecialmodeltrainstatuschangenotifyreq)| 否 |none|

> 返回示例

> 200 Response

```json
{
  "success": false,
  "code": 0,
  "msg": "",
  "data": ""
}
```

### 返回结果

|状态码|状态码含义|说明|数据模型|
|---|---|---|---|
|200|[OK](https://tools.ietf.org/html/rfc7231#section-6.3.1)|none|[SingleResponseString](#schemasingleresponsestring)|

# 数据模型

<h2 id="tocS_SpecialModelTrainStatusChangeNotifyReq">SpecialModelTrainStatusChangeNotifyReq</h2>

<a id="schemaspecialmodeltrainstatuschangenotifyreq"></a>
<a id="schema_SpecialModelTrainStatusChangeNotifyReq"></a>
<a id="tocSspecialmodeltrainstatuschangenotifyreq"></a>
<a id="tocsspecialmodeltrainstatuschangenotifyreq"></a>

```json
{
  "trainTaskId": "string",
  "statusType": 0,
  "failureMessage": "string"
}

```

### 属性

|名称|类型|必选|约束|中文名|说明|
|---|---|---|---|---|---|
|trainTaskId|string|true|none||任务ID（必传）|
|statusType|integer|true|none||状态类型（必传，1-排队，2-训练中，3-训练完成，4-训练失败 5-取消）|
|failureMessage|string|false|none||失败原因（状态为异常或取消时必传，500字符内）|

<h2 id="tocS_SingleResponseString">SingleResponseString</h2>

<a id="schemasingleresponsestring"></a>
<a id="schema_SingleResponseString"></a>
<a id="tocSsingleresponsestring"></a>
<a id="tocssingleresponsestring"></a>

```json
{
  "success": true,
  "code": 0,
  "msg": "string",
  "data": "string"
}

```

### 属性

|名称|类型|必选|约束|中文名|说明|
|---|---|---|---|---|---|
|success|boolean|false|none||none|
|code|integer|false|none||返回编码|
|msg|string|false|none||返回消息|
|data|string|false|none||返回单个对象|
