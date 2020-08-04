rm -rf ./*.whl
pip download --no-deps \
    pydantic==1.6.1 \
    timm==0.1.30 \
    pytorch-argus==0.1.1

rm -rf ./argus-birdsong
git clone https://github.com/lRomul/argus-birdsong
cp -r "/workdir/data/experiments/$1" "./argus-birdsong/data/experiments/$1"
