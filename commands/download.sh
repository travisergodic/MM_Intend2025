if [ ! -d "/content" ]; then
    mkdir /content
    cd /content
else
    echo "/content directory already exists. Using the existing directory."
    cd /content
fi

pip install gdown
gdown https://drive.google.com/uc?id=1v43kzCzWbsEkPtrAw1d0xSSAyXTYbgNT -O /content/train_data.zip
gdown https://drive.google.com/uc?id=1OnWa9TqmpSmC8lzWkQEAh9E6xLngZnM6 -O /content/round1_test_data.zip
# gdown https://drive.google.com/uc?id=1SVeDiHSL_Qj9HesVVfMfRoZlj3UOuRET -O /content/train_resize.zip
# gdown https://drive.google.com/uc?id=1Wdepa-pCrfx0d4jw4pQvlsavEgyPR0mr -O /content/test1_resize.zip
# gdown https://drive.google.com/uc?id=1r0Z7we3kuL_YW8ygTZ9PcQ0bx2Us-p2T -O /content/train_intend_resize_560.zip
# gdown https://drive.google.com/uc?id=1BUMN__ZCl8V6UEIdnOtLMmwNgVBJ0LJS -O /content/test1_intend_resize_560.zip
# gdown https://drive.google.com/uc?id=1AwE4LNvRsAdBjkc5Es5CoPKwi9dbPsxk -O /content/train_resize_313600.zip
# gdown https://drive.google.com/uc?id=1wuc4nJSeitxApaPqDeG1od6-6VmKt_MG -O /content/test1_resize_313600.zip
# gdown https://drive.google.com/uc?id=1kncmu4jXZg6pbTwVFEZki2qZMMRUrQ6j -O /content/crop_scene.zip

gdown https://drive.google.com/uc?id=1fPAAJpKYEdF41AYb43W7aZo7fu9XvD2q
gdown https://drive.google.com/uc?id=1BXR59txOVvX89yrVt3_rkLBwCJW0nNEe
gdown https://drive.google.com/uc?id=1iWZtSbDxr8b1OG6Y1E14a6ji24yOpQ10
gdown https://drive.google.com/uc?id=1ZqyZwmJeSmYJ_qj_SdNRv7fkFXekCx2r

# unzip
unzip train_data.zip
unzip round1_test_data.zip
unzip test1_resize_intend_313600.zip
unzip test1_resize_intend_616.zip
unzip train_resize_intend_313600.zip
unzip train_resize_intend_616.zip
# unzip train_intend_resize_560.zip
# unzip test1_intend_resize_560.zip
# unzip train_resize_313600.zip
# unzip test1_resize_313600.zip
# unzip crop_scene.zip