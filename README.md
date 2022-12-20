# Deep Learning Kelompok 1

1. Ida Ayu Oka Dewi Cahyani (1905551146) - ([@dycahyani](https://www.github.com/dycahyani))
2. I Made Andre Dwi Winama Putra (1905551003) - ([@andre002wp](https://www.github.com/andre002wp))
3. I Putu Kevin Ari Narayana (1905551030) - ([@arve1n](https://www.github.com/arve1n))
4. Luh Putu Murniasih Pertiwi (1905551038) - ([@murniasihpertiwi](https://www.github.com/murniasihpertiwi))

# Real Time Face Mask Detection berbasis Desktop Menggunakan Model CNN

## Deskripsi Aplikasi
Face mask detection adalah pendeteksian apakah seseorang menggunakan masker atau tidak menggunakan masker. Dimana kami menggunakan metode Convolutional Neural Networks (CNN). Algoritma pembelajaran Convolutional Neural Networks memanfaatkan ekstraksi fitur dari citra yang nanti akan dipelajari oleh beberapa hidden layer. Sistem ini menggunakan kombinasi klasifikasi deteksi objek, gambar, dan pelacakan objek sehingga dapat mengembangkan sistem yang mendeteksi wajah bermasker atau tidak bermasker dalam gambar atau video secara realtime. Dataset yang diambil bervariasi dengan gambar wajah menggunakan hijab, topi dan tidak menggunakan atribut. Selain itu, gambar yang diambil dari berbagai negara seperti asia, eropa dan amerika.

## Arsitektur Aplikasi
Deep Learning Architecture mempelajari hal penting yaitu fitur non-linier dari sampel yang diberikan. Kemudian, mempelajari arsitektur yang digunakan untuk memprediksi sampel yang sebelumnya tidak terlihat. Deep Learning Architecture sangat tergantung pada CNN. Pada Deep Learning Architecture ada beberapa tahapan yang harus dilakukan yaitu, Dataset Collection, Training, dan Deployment. 

![model arsitektur](https://user-images.githubusercontent.com/79149921/208701773-0c4224d8-d6f1-496f-a164-9b8861e0d720.png)

## Dataset
Dataset yang digunakan adalah data yang diambil dari Kaggle yaitu data Face Mask Detection milik Vijay Kumar. Data yang dikumpulkan terdapat 3 jenis class, yaitu with mask, without mask, dan mask weared incorrect. Dari data yang telah dikumpulkan diperoleh total 8982 sampel.

## Metode Penelitian

### Data Preparation
Pada tahapan ini, akan dilakukan pengolahan data menggunakan data yang telah dikumpulkan. Data yang dimaksud adalah dari Kaggle yaitu data Face Mask Detection milik Vijay Kumar, data yang dikumpulkan terdapat 3 jenis class, yaitu with mask, without mask, dan mask weared incorrect.

### Data Preprocessing
Data preprocessing merupakan beberapa proses persiapan data sebelum dilakukan proses training untuk pembuatan model. 
1) Data Selection
Proses ini dilakukan untuk menganilisis data-data yang relevan karena sering ditemukan bahwa tidak semua data dibutuhkan. Pada penelitian ino kami mengunakan semua data pada dataset yaitu gambar wajah memakai masker dengan label with mask, gambar wajah tidak memakai masker dengan label without mask, dan gambar wajah dengan penggunaan masker yang salah dengan label mask weared incorrect.
2) Data Cleaning
Hal ini dilakukan agar tidak ada duplikasi data sehingga data tersebut dapat diolah dan dilakukan proses pembuatan model.
3) Face Cropping
Pembagian pada bagian wajah ini terdiri dari 3 (tiga) label yaitu bagian wajah yang memakai masker, bagian wajah yang tidak memakai masker, serta wajah dengan penggunaan masker yang salah. Hal ini bertujuan agar saat proses pengolahan dataset menjadi model hanya bagian wajah saja yang kan dilakukan proses training sehingga bagian selain wajah tidak perlu dilakukan pencocokan. Proses face cropping ini menggunakan algoritma Yolov5-face smallest weights sebagai pendeteksi wajahnya.
4) Resize Ukuran Image pada Dataset
Pada tahap ini akan dilakukan pengolahan dataset dengan cara mengecilkan atau mengatur image size dari dataset yang telah dikumpulkan. Hal ini bertujuan agar saat penginputan dan proses classification pada arsitektur Fully CNN menjadi seragam dan mengatasi loss accuracy atau kehilangan tingkat akurasi pada proses training. Pada proses ini image akan di resize menajadi ukuran 128 x 128 pixel.

### Membuat Model Fully CNN
Convolutional Neural Network (CNN) adalah salah satu jenis neural network yang biasa digunakan pada data image. CNN bisa digunakan untuk mendeteksi dan mengenali object pada sebuah image. CNN adalah sebuah teknik yang terinspirasi dari cara mamalia dan manusia dapat menghasilkan persepsi visual.
Secara garis besar Convolutional Neural Network (CNN) tidak jauh beda dengan neural network biasanya. CNN terdiri dari neuron yang memiliki weight, bias dan activation function. Convolutional layer juga terdiri dari neuron yang tersusun sedemikian rupa sehingga membentuk sebuah filter dengan panjang dan tinggi (pixels).

Pada model deep transfer learning akan terdapat beberapa lapisan (layer) untuk mengenali sebuah objek. Lapisan pertama dan tengah model bertanggung jawab untuk mengenali bentuk dalam gambar. Misalnya, lapisan pertama dalam model pengenalan wajah manusia dapat mengenali garis, lapisan kedua lingkaran, mata lapisan ketiga, dan wajah lapisan keempat. Sekarang lapisan telah dilatih, mereka dapat digunakan dalam kumpulan data lain, yang pada penelitian ini adalah penggunaan masker wajah. Gambar berikut menunjukkan flowchartmodel yang diusulkan.

![image](https://user-images.githubusercontent.com/79149921/207069065-26cb01ff-6afa-4a45-9f9a-0064a9761270.png)


### Proses Training
Training merupakan processing yang terfokus untuk memuat dataset face mask detection dari penyimpanan dataset, melatih model dengan mengunakan instrument dari Keras atau TensorFlow dari dataset ini, akan membuat serial face mask detection pada penyimpanan dataset. Berikut merupakan langkah-langkah yang dilakukan pada proses training.

#### Face Segmentation

1. Download Yolov5-face smallest weights
2. Load Model
3. Extract faces from image
4. Get bounding box

#### Proses Training Model Mask Detection
1. Pengecekan jumlah data pada setiap class

![image](https://user-images.githubusercontent.com/79149921/207525158-f6bf184b-8c2a-4713-b383-a16cd187de2e.png)

Jumlah foto setiap class sudah sama yaitu masing-masing sebanyak 2994 data, sehingga data siap memasuki tahap augmentasi.

2. Augmentasi Data
Augemntasi gambar adalah teknik yang berguna untuk memperluas data pelatihan model tanpa perlu mencari data tambahan. Augmentasi gambar, adalah tindakan mereplika gambar yang ada dengan berbagai penyesuaian untuk memperbanyak data latih. Model terlatih akan lebih realistis dari kondisi dunia nyata dan akan mampu beradaptasi dengan berbagai perubahan kondisi yang ada.

![image](https://user-images.githubusercontent.com/79149921/207525401-df72a39c-bb8d-486d-b183-a179da0b7acf.png)

Output Augmentasi :
- Training = 5748 Gambar untuk 3 Class
- Validation = 1437 Gambar untuk 3 Class
- Test = 1797 Gambar untuk 3 Class

3. Modelling Data

![image](https://user-images.githubusercontent.com/79149921/207525666-1f3aa693-092d-4103-87a3-e07986ea3394.png)

4. Training Model

![image](https://user-images.githubusercontent.com/79149921/207525967-44479d98-18ce-481b-a2a9-ad3e5d17c0fb.png)
![image](https://user-images.githubusercontent.com/79149921/207526030-d7d3fca6-d0e5-46a9-906b-cb6b766cb76c.png)

- Compiler = Adams
- Metrics= Accuracy
- Jumlah Epoch= 50 Epochs

### Proses Testing
Proses Testing data menggunakan 6 jenis model yang berbeda diantaranya 4 model CNN lalu Resnet50 dan EfficientNetB0. Berikut ini merupakan perbandingan dari masing-masing model yang telah dicoba.

#### CNN Model

![image](https://user-images.githubusercontent.com/79149921/207531794-54924ab4-686d-4da5-b8c6-453cdbc90e5b.png)

Hasil Test Set
- Loss = 0.1544
- Accuracy = 0.9694

#### Fully CNN

![image](https://user-images.githubusercontent.com/79149921/207532083-b26849ab-8b58-4aad-8cc6-fc0c12e5d303.png)  ![image](https://user-images.githubusercontent.com/79149921/207532198-45798084-93f6-48ef-a24c-07a1415d8f95.png)

Akurasi : 0.9129

#### CNN Model 2

![image](https://user-images.githubusercontent.com/79149921/207532418-6b4e4298-b7a9-4100-bc87-157bb79df6e2.png)  ![image](https://user-images.githubusercontent.com/79149921/207532535-1e2c3770-2030-4d7d-a98d-a4feecb1b2b1.png)

Akurasi : 0.8896

#### CNN Model 3

![image](https://user-images.githubusercontent.com/79149921/207532764-bd4c03b4-c0fb-42fc-b3cc-833c655ca78f.png)  ![image](https://user-images.githubusercontent.com/79149921/207532850-13473646-dbd7-4a32-b91c-ce34daeff5cd.png)

Akurasi : 0.8883

#### EffNet

![image](https://user-images.githubusercontent.com/79149921/207532999-75235292-d517-47e2-8976-a75b77850fc3.png)  ![image](https://user-images.githubusercontent.com/79149921/207533080-b77c2db0-86fb-4efe-a04e-eca5ded6f833.png)

Akurasi : 0.9472

#### ResNett

![image](https://user-images.githubusercontent.com/79149921/207533317-7c6bd705-a5b9-4337-a114-f9da64ab3b09.png)  ![image](https://user-images.githubusercontent.com/79149921/207533424-55bd60c6-0c7a-42d2-968a-a9a57b7dcbd7.png)

Akurasi : 0.90


### Proses Evaluasi
(penjelasan)

## Cara Menjalankan Aplikasi
= Command Interface
1. Download dan extract isi project kedalam 1 folder
2. Download yolov5-face pretrain model yang ingin digunakan pada https://github.com/deepcam-cn/yolov5-face (disarankan menggunakan model yang kecil)
3. Letakkan yolo model yang telah didowload pada folder Yolo
4. Konfigurasikan sistem yolo yang ingin digunakan pada bagian attempt_load line 39 (secara default akan digunakan model terkecil yang ditemukan dalam folder yolo)

![image](https://user-images.githubusercontent.com/24908637/206091718-36354f82-89c3-4798-a588-6cce0915d04e.png)

5. Dengan membuka folder aplikasi, jalankan file mask_detection_inference.py dengan perintah "py mask_detection_inference.py" pastikan aplikasi dapat membaca asset yang dibutuhkan.

