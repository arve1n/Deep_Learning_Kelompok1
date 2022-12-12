# Deep Learning Kelompok 1

1. Ida Ayu Oka Dewi Cahyani (1905551146) - ([@dycahyani](https://www.github.com/dycahyani))
2. I Made Andre Dwi Winama Putra (1905551003) - ([@andre002wp](https://www.github.com/andre002wp))
3. I Putu Kevin Ari Narayana (1905551030) - ([@arve1n](https://www.github.com/arve1n))
4. Luh Putu Murniasih Pertiwi (1905551038) - ([@murniasihpertiwi](https://www.github.com/murniasihpertiwi))

# Real Time Face Mask Detection berbasis ...

## Deskripsi Aplikasi
Face mask detection adalah pendeteksian apakah seseorang menggunakan masker atau tidak menggunakan masker. Dimana kami menggunakan metode Convolutional Neural Networks (CNN). Algoritma pembelajaran Convolutional Neural Networks memanfaatkan ekstraksi fitur dari citra yang nanti akan dipelajari oleh beberapa hidden layer. Sistem ini menggunakan kombinasi klasifikasi deteksi objek, gambar, dan pelacakan objek sehingga dapat mengembangkan sistem yang mendeteksi wajah bermasker atau tidak bermasker dalam gambar atau video secara realtime. Dataset yang diambil bervariasi dengan gambar wajah menggunakan hijab, topi dan tidak menggunakan atribut. Selain itu, gambar yang diambil dari berbagai negara seperti asia, eropa dan amerika.

## Arsitektur Aplikasi
Deep Learning Architecture mempelajari hal penting yaitu fitur non-linier dari sampel yang diberikan. Kemudian, mempelajari arsitektur yang digunakan untuk memprediksi sampel yang sebelumnya tidak terlihat. Deep Learning Architecture sangat tergantung pada CNN. Pada Deep Learning Architecture ada beberapa tahapan yang harus dilakukan yaitu, Dataset Collection, Training, dan Deployment. 

## Dataset
Dataset yang digunakan adalah data yang diambil dari Kaggle yaitu data Face Mask Detection milik Vijay Kumar. Data yang dikumpulkan terdapat 3 jenis class, yaitu with mask, without mask, dan mask weared incorrect. Dari data yang telah dikumpulkan diperoleh total 8982 sampel.

## Metode Penelitian

### Data Preparation
Pada tahapan ini, akan dilakukan pengolahan data menggunakan data yang telah dikumpulkan. Data yang dimaksud adalah dari Kaggle yaitu data Face Mask Detection milik Vijay Kumar, data yang dikumpulkan terdapat 3 jenis class, yaitu with mask, without mask, dan mask weared incorrect.

### Data Preprocessing
Data preprocessing merupakan beberapa proses persiapan data sebelum dilakukan proses training untuk pembuatan model Deep Learning. 
1) Data Selection
Proses ini dilakukan untuk menganilisis data-data yang relevan karena sering ditemukan bahwa tidak semua data dibutuhkan. Pada penelitian ino kami mengunakan semua data pada dataset yaitu gambar wajah memakai masker dengan label with mask, gambar wajah tidak memakai masker dengan label without mask, dan gambar wajah dengan penggunaan masker yang salah dengan label mask weared incorrect.
2) Data Cleaning
Hal ini dilakukan agar tidak ada duplikasi data sehingga data tersebut dapat diolah dan dilakukan proses pembuatan model.
3) Face Cropping
Pembagian pada bagian wajah ini terdiri dari 3 (tiga) label yaitu bagian wajah yang memakai masker, bagian wajah yang tidak memakai masker, serta wajah dengan penggunaan masker yang salah. Hal ini bertujuan agar saat proses pengolahan dataset menjadi model hanya bagian wajah saja yang kan dilakukan proses training sehingga bagian selain wajah tidak perlu dilakukan pencocokan. Proses face cropping ini menggunakan algoritma Haar Cascade sebagai pendeteksi wajahnya.
4) Resize Ukuran Image pada Dataset
Pada tahap ini akan dilakukan pengolahan dataset dengan cara mengecilkan atau mengatur image size dari dataset yang telah dikumpulkan. Hal ini bertujuan agar saat penginputan dan proses classification pada arsitektur MobileNetV2 menjadi seragam dan mengatasi loss accuracy atau kehilangan tingkat akurasi pada proses training. Pada proses ini image akan di resize menajadi ukuran 224 x 224 pixel.

### Membuat Model
(ini contoh penjelasan aja) Model yang digunakan pada penelitian ini menggunakan arsitektur deep transfer learning. Transfer learning pada bidang computer vision
didasarkan pada premis bahwa model yang dilatih pada kumpulan data besar dari gambar yang tersedia dapat digunakan sebagai model dasar untuk mengenali fitur atau bentuk objek di dunia nyata. Melalui transfer learning memungkinkan untukmenggunakan fitur ini tanpa melatih ulang model dari awal [16].

Pada model deep transfer learning akan terdapat beberapa lapisan (layer) untuk mengenali sebuah objek. Lapisan pertama dan tengah model bertanggung jawab untuk mengenali bentuk dalam gambar. Misalnya, lapisan pertama dalam model pengenalan wajah manusia dapat mengenali garis, lapisan kedua lingkaran, mata lapisan ketiga, dan wajah lapisan keempat. Sekarang lapisan telah dilatih, mereka dapat digunakan dalam kumpulan data lain, yang pada penelitian ini adalah penggunaan masker wajah. Gambar berikut menunjukkan flowchartmodel yang diusulkan.

![image](https://user-images.githubusercontent.com/79149921/207069065-26cb01ff-6afa-4a45-9f9a-0064a9761270.png)


### Proses Training
Training merupakan processing yang terfokus untuk memuat dataset face mask detection dari penyimpanan dataset, melatih model dengan mengunakan instrument dari Keras atau TensorFlow dari dataset ini, akan membuat serial face mask detection pada penyimpanan dataset. 

### Proses Testing

### Proses Evaluasi

## Cara Menjalankan Aplikasi
1. Download yolov5-face pretrain model yang ingin digunakan pada https://github.com/deepcam-cn/yolov5-face (disarankan menggunakan model yang kecil)
2. Letakkan yolo model yang telah didowload pada folder Yolo
3. Konfigurasikan sistem yolo yang ingin digunakan pada bagian attempt_load line 39 (secara default akan digunakan model terkecil yang ditemukan dalam folder yolo)

![image](https://user-images.githubusercontent.com/24908637/206091718-36354f82-89c3-4798-a588-6cce0915d04e.png)

