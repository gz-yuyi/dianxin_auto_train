
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

## POST 专用模型训练-发布完成结果通知接口

POST /api/model/train/notify/publish_result

> Body 请求参数

```json
{
  "trainTaskId": 0,
  "publishResult": 0,
  "failureMessage": "string"
}
```

### 请求参数

|名称|位置|类型|必选|说明|
|---|---|---|---|---|
|body|body|[SpecialModelPublishResultNotifyReq](#schemaspecialmodelpublishresultnotifyreq)| 否 |none|

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

<h2 id="tocS_SpecialModelTrainResultNotifyReq">SpecialModelTrainResultNotifyReq</h2>

<a id="schemaspecialmodeltrainresultnotifyreq"></a>
<a id="schema_SpecialModelTrainResultNotifyReq"></a>
<a id="tocSspecialmodeltrainresultnotifyreq"></a>
<a id="tocsspecialmodeltrainresultnotifyreq"></a>

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

### 属性

|名称|类型|必选|约束|中文名|说明|
|---|---|---|---|---|---|
|trainTaskId|integer(int64)|true|none||任务ID（必传）|
|epoch|integer|true|none||轮次（必传）|
|accuracy|number|false|none||正确率|
|loss|number|false|none||损失率|
|valAccuracy|number|false|none||验证正确率|
|valLoss|number|false|none||验证损失率|
|f1Score|number|false|none||f1分数|

<h2 id="tocS_SpecialModelPublishResultNotifyReq">SpecialModelPublishResultNotifyReq</h2>

<a id="schemaspecialmodelpublishresultnotifyreq"></a>
<a id="schema_SpecialModelPublishResultNotifyReq"></a>
<a id="tocSspecialmodelpublishresultnotifyreq"></a>
<a id="tocsspecialmodelpublishresultnotifyreq"></a>

```json
{
  "trainTaskId": 0,
  "publishResult": 0,
  "failureMessage": "string"
}

```

### 属性

|名称|类型|必选|约束|中文名|说明|
|---|---|---|---|---|---|
|trainTaskId|integer(int64)|true|none||任务ID（必传）|
|publishResult|integer|true|none||发布结果（必传，1-成功，0-失败）|
|failureMessage|string|false|none||发布失败原因（如失败，100字符内）|

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