## Expanding CENet for panoptic segmentation

### Environment

Build docker
1. docker-compose -f ./docker-compose.yaml build

Run docker
2. docker-compose -f ./docker-compose.yaml run --rm workspace

### Evaluation

Semantic inference authors model:

- `python3 infer.py -d /data/ -l preds_semantic_cenet_authors/ -m Final\ result/512-594/ -s valid`


