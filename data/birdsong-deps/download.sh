cd $(dirname "${BASH_SOURCE[0]}")

rm -rf ./*.whl
pip download --no-deps \
    pydantic==1.6.1 \
    timm==0.1.30 \
    pytorch-argus==0.1.1 \
    resnest==0.0.5

rm -rf ./argus-birdsong
git clone https://github.com/lRomul/argus-birdsong && cd argus-birdsong && git checkout "$2" && cd ..

cp -r "/workdir/data/experiments/$1" "./argus-birdsong/data/experiments/$1"
