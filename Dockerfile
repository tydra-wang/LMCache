# The vLLM Dockerfile is used to construct vLLM image that can be directly used
# to run the OpenAI compatible server.

# Please update any changes made here to
# docs/source/contributing/dockerfile/dockerfile.md and
# docs/source/assets/contributing/dockerfile-stages-dependency.png

ARG CUDA_VERSION=12.4.1

#################### vLLM installation IMAGE ####################
# image with vLLM installed
# TODO: Restore to base image after FlashInfer AOT wheel fixed
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04 AS vllm-base
ARG CUDA_VERSION=12.4.1
ARG PYTHON_VERSION=3.12
WORKDIR /vllm-workspace
ENV DEBIAN_FRONTEND=noninteractive
ARG TARGETPLATFORM

RUN PYTHON_VERSION_STR=$(echo ${PYTHON_VERSION} | sed 's/\.//g') && \
    echo "export PYTHON_VERSION_STR=${PYTHON_VERSION_STR}" >> /etc/environment

# Install minimal dependencies and uv
RUN apt-get update -y \
    && apt-get install -y ccache git curl wget sudo vim \
    && apt-get install -y ffmpeg libsm6 libxext6 libgl1 libibverbs-dev \
    && curl -LsSf https://astral.sh/uv/install.sh | sh

# Add uv to PATH
ENV PATH="/root/.local/bin:$PATH"
# Create venv with specified Python and activate by placing at the front of path
ENV VIRTUAL_ENV="/opt/venv"
RUN uv venv --python ${PYTHON_VERSION} --seed ${VIRTUAL_ENV}
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# This timeout (in seconds) is necessary when installing some dependencies via uv since it's likely to time out
# Reference: https://github.com/astral-sh/uv/pull/1694
ENV UV_HTTP_TIMEOUT=500

# Workaround for https://github.com/openai/triton/issues/2507 and
# https://github.com/pytorch/pytorch/issues/107960 -- hopefully
# this won't be needed for future versions of this docker image
# or future versions of triton.
RUN ldconfig /usr/local/cuda-$(echo $CUDA_VERSION | cut -d. -f1,2)/compat/

# arm64 (GH200) build follows the practice of "use existing pytorch" build,
# we need to install torch and torchvision from the nightly builds first,
# pytorch will not appear as a vLLM dependency in all of the following steps
# after this step
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ "$TARGETPLATFORM" = "linux/arm64" ]; then \
        uv pip install --index-url https://download.pytorch.org/whl/nightly/cu124 "torch==2.6.0.dev20241210+cu124" "torchvision==0.22.0.dev20241215";  \
    fi

# Install vllm wheel first, so that torch etc will be installed.
ARG VLLM_COMMIT=10f755278953adec6c8e0916c0b2f636b4dd21e8
RUN --mount=type=cache,target=/root/.cache/uv uv pip install vllm --extra-index-url https://wheels.vllm.ai/${VLLM_COMMIT}

# If we need to build FlashInfer wheel before its release:
# $ export FLASHINFER_ENABLE_AOT=1
# $ # Note we remove 7.0 from the arch list compared to the list below, since FlashInfer only supports sm75+
# $ export TORCH_CUDA_ARCH_LIST='7.5 8.0 8.6 8.9 9.0+PTX'
# $ git clone https://github.com/flashinfer-ai/flashinfer.git --recursive
# $ cd flashinfer
# $ git checkout 524304395bd1d8cd7d07db083859523fcaa246a4
# $ rm -rf build
# $ python3 setup.py bdist_wheel --dist-dir=dist --verbose
# $ ls dist
# $ # upload the wheel to a public location, e.g. https://wheels.vllm.ai/flashinfer/524304395bd1d8cd7d07db083859523fcaa246a4/flashinfer_python-0.2.1.post1+cu124torch2.5-cp38-abi3-linux_x86_64.whl

# RUN --mount=type=cache,target=/root/.cache/uv \
# if [ "$TARGETPLATFORM" != "linux/arm64" ]; then \
#     uv pip install https://github.com/flashinfer-ai/flashinfer/releases/download/v0.2.1.post1/flashinfer_python-0.2.1.post1+cu124torch2.5-cp38-abi3-linux_x86_64.whl ; \
# fi
# COPY examples examples

# Although we build Flashinfer with AOT mode, there's still
# some issues w.r.t. JIT compilation. Therefore we need to
# install build dependencies for JIT compilation.
# TODO: Remove this once FlashInfer AOT wheel is fixed
# COPY requirements/build.txt requirements/build.txt
# RUN --mount=type=cache,target=/root/.cache/uv \
#     uv pip install -r requirements/build.txt


ARG torch_cuda_arch_list='7.0 7.5 8.0 8.6 8.9 9.0+PTX'
ENV TORCH_CUDA_ARCH_LIST=${torch_cuda_arch_list}
ARG vllm_fa_cmake_gpu_arches='80-real;90-real'
ENV VLLM_FA_CMAKE_GPU_ARCHES=${vllm_fa_cmake_gpu_arches}

#RUN --mount=type=bind,target=/lmcache \
#    --mount=type=cache,target=/root/.cache/uv \
#    TORCH_CUDA_ARCH_LIST=${torch_cuda_arch_list} VLLM_FA_CMAKE_GPU_ARCHES=${vllm_fa_cmake_gpu_arches} uv pip install -e /lmcache

RUN --mount=type=cache,target=/root/.cache/uv \
    git clone https://github.com/LMCache/LMCache.git /tmp/LMCache && uv pip install -e /tmp/LMCache

#################### vLLM installation IMAGE ####################

#################### OPENAI API SERVER ####################
# base openai image with additional requirements, for any subsequent openai-style images
FROM vllm-base AS vllm-openai-base

# This timeout (in seconds) is necessary when installing some dependencies via uv since it's likely to time out
# Reference: https://github.com/astral-sh/uv/pull/1694
ENV UV_HTTP_TIMEOUT=500

# install additional dependencies for openai api server
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ "$TARGETPLATFORM" = "linux/arm64" ]; then \
        uv pip install accelerate hf_transfer 'modelscope!=1.15.0' 'bitsandbytes>=0.42.0' 'timm==0.9.10' boto3 runai-model-streamer runai-model-streamer[s3]; \
    else \
        uv pip install accelerate hf_transfer 'modelscope!=1.15.0' 'bitsandbytes>=0.45.0' 'timm==0.9.10' boto3 runai-model-streamer runai-model-streamer[s3]; \
    fi

ENV VLLM_USAGE_SOURCE production-docker-image

# define sagemaker first, so it is not default from `docker build`
FROM vllm-openai-base AS vllm-sagemaker

COPY examples/online_serving/sagemaker-entrypoint.sh .
RUN chmod +x sagemaker-entrypoint.sh
ENTRYPOINT ["./sagemaker-entrypoint.sh"]

FROM vllm-openai-base AS vllm-openai

ENTRYPOINT ["python3", "-m", "vllm.entrypoints.openai.api_server"]
#################### OPENAI API SERVER ####################

