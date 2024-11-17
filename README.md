# Improving SelfRefromer model with MCTF(MCTF를 사용한 SelfReformer 경량화 연구)

다음의 연구는 vit기반으로 개발된 SelfReformer 모델을 token fusion의 최신 기법 중 하나인 MCTF기법을 사용하여 경량화하였다.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)

## Introduction
SOD(Salient Object Detection): 이미지 또는 비디오에서 사람의 이목을 끌 확률이 가장 높은 물체(=주요 물체)를 검출하는 기법

![task-0000000742-bab03b67](https://github.com/user-attachments/assets/b5830c36-4ae8-4768-8e13-beed002bd345)
(출처: https://paperswithcode.com/task/salient-object-detection)

SelfReformer(https://arxiv.org/pdf/2205.11283)는 Pyramid-ViT 기반의 SOD 모델이다.

![image](https://github.com/user-attachments/assets/ce067397-7665-4c1d-ace3-337cd6551b77)
(출처: SelfReformer: Self-Refined Network with Transformer for Salient Object Detection https://arxiv.org/pdf/2205.11283)

본 연구는 이 모델을 transformer 모델 경량화 기법을 사용하여 더 적은 Gflops로 정확도를 유지하는 경량화 모델을 개발하였다.

사용한 경량화 기법은 MCTF(https://github.com/mlvlab/MCTF)기법을 사용한다.

## Features
![image](https://github.com/user-attachments/assets/9ead283f-cec3-4690-b864-2c39c392e867)<br/>
가장 많은 작업을 필요로 하는 Local Context Branch에 중점적으로 경량화 레이어를 삽입하였다.

R = Reduced tokens<br/>
stage 1: 49 -> 36, R = 13<br/>
stage 2: 196 -> 144, R = 53<br/>
stage 3: 784 -> 576, R = 208<br/>
stage 4: 3136 -> 2304, R = 832

<Gflops>
Before MCTF: 21.695107798<br/>
After MCTF: 16.715654545

<MAE>
Before MCTF: 0.0274<br/>
After MCTF: 0.0402

실험 결과 original model에서 큰 정확도 손실 없이 Gflops를 감소시키는 효과를 볼 수 있었다.

![image](https://github.com/user-attachments/assets/ac62c3bb-c53f-444f-8e2c-4e49bfdf1e1d)

## Installation
