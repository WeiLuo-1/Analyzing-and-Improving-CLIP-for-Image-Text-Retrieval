# Results Summary

|Method|Split|Max Examples|Prompt|Rerank|I2T R@1|I2T R@5|I2T R@10|T2I R@1|T2I R@5|T2I R@10|
|---|---|---|---|---|---|---|---|---|---|---|
|Baseline|test|200|none|none|0.930|0.980|0.995|0.791|0.948|0.976|
|Baseline (200)|test|200|none|none|0.930|0.980|0.995|0.791|0.948|0.976|
|Baseline (2000)|test|2000|none|none|0.726|0.922|0.964|0.501|0.769|0.849|
|Photo+Image Prompt|test|200|photo_image|none|0.920|0.985|0.995|0.784|0.950|0.978|
|Photo+Image Prompt (2000)|test|2000|photo_image|none|0.745|0.924|0.966|0.504|0.768|0.849|
|Detailed Prompt|test|200|detailed_scene|none|0.925|0.975|0.995|0.777|0.953|0.977|
|Detailed Prompt (2000)|test|2000|detailed_scene|none|0.734|0.927|0.963|0.489|0.759|0.841|
|Rerank|test|200|none|caption_prior|0.930|0.980|0.995|0.791|0.952|0.980|
|Rerank (2000)|test|2000|none|caption_prior|0.726|0.922|0.964|0.511|0.784|0.862|
|Smoke Test|test|50|none|none|0.940|0.980|1.000|0.888|0.976|0.984|
