# Install QNN and Hexagon SDK
:last update 2025-03-25:

There are various ways to install qnn.
In this doc, we recomend you to use qpm-cli.
If you install the QNN and Hexagon SDK for NNtrainer usage, please follow the versions of:

- QNN (a.k.a., Qualcoomm Neuarl Processing SDK) version 2.31.0.250130
- HexagonSDK version 5.5.2.0

## Prepare qpm-cli


1. Download qpm-cli

> https://qpm.qualcomm.com/#/main/tools/details/QPM3
2. Install qpm-cli

> sudo apt-get install ./QualcommPackageManager3.3.0.117.0.Linux-x86.deb
- In order to use qpm-cli, you may need to login
```
$ qpm-cli login
```

## Install QNN v 2.31

Install qnn version 2.31

```
$ qpm-cli --product-list | grep neural
    qualcomm_neural_processing_sdk
    qualcomm_neural_processing_sdk_public
$ qpm-cli --install qualcomm_neural_processing_sdk --version 2.31.0.250130
```

- Please be sure where the qnn is installed.
- It would be `/opt/qcom/aistack/qairt/2.31.0.250130` by default.

## Install Hexagon SDK 5.5.2.0

```
$ qpm-cli --license-activate hexagonsdk5.x
$ qpm-cli --install hexagonsdk5.x --version 5.5.2.0
[Error] : Required dependency criteria not met. HexagonSDK6.x should be installed before installing Compute1.x
[Error] : Required version of component HexagonSDK6.x.Core is not installed on the machine
[Warning] : Compute1.x.1.12.0.Linux-x64.qik was not installed. Reason: ErrorProcessingComponents
[Info] : SUCCESS: Installed HexagonSDK5.x.Core at /local/mnt/workspace/Qualcomm/Hexagon_SDK/5.5.2.0
```

### Trouble shooting

1. Access to the path is denied

Phenomenon:
```
[Info] : Extracting files
[Fatal] : Access to the path '/local/mnt/workspace/Qualcomm/Hexagon_SDK/5.5.2.0' is denied.
[Error] : Installation failed with Exception
```

Trouble-shoot:
```
$ mkdir -p /local/mnt/workspace/Qualcomm/
$ sudo chmod 777 /local/mnt/workspace/Qualcomm/
```

2. error while loading shared libraries: libtinfo.so.5: cannot open shared object file: No such file or directory
```
sudo apt install libncurses5
```

