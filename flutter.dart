import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'dart:async';


void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Jetson Nano Park App',
      theme: ThemeData.dark().copyWith(
        primaryColor: Colors.blue,
        colorScheme: ColorScheme.dark().copyWith(
          secondary: Colors.tealAccent,
        ),
        scaffoldBackgroundColor: Colors.grey[800],
        appBarTheme: AppBarTheme(
          backgroundColor: Color.fromARGB(255, 118, 185, 0),
        ),
        elevatedButtonTheme: ElevatedButtonThemeData(
          style: ButtonStyle(
            foregroundColor: MaterialStateProperty.all<Color>(Colors.white),
          ),
        ),
      ),
      home: BaseUrlScreen(),
    );
  }
}

class BaseUrlScreen extends StatelessWidget {
  final TextEditingController baseUrlController = TextEditingController();
  final _formKey = GlobalKey<FormState>();

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Jetson Nano Park App'),
      ),
      body: Center(
        child: Form(
          key: _formKey,
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Padding(
                padding: const EdgeInsets.all(8.0),
                child: TextFormField(
                  controller: baseUrlController,
                  decoration: const InputDecoration(
                    border: OutlineInputBorder(),
                    labelText: 'IP Adresini Giriniz (ör: 192.168.196.113)',
                  ),
                  style: TextStyle(fontSize: 25),
                  validator: (value) {
                    if (value == null || value.isEmpty) {
                      return 'Lütfen IP Girin';
                    }
                    return null;
                  },
                ),
              ),
              ElevatedButton(
                onPressed: () {
                  if (_formKey.currentState?.validate() ?? false) {
                    Navigator.push(
                      context,
                      MaterialPageRoute(
                        builder: (context) => ConnectionScreen(
                          baseUrl: baseUrlController.text,
                        ),
                      ),
                    );
                  }
                },
                child: const Text('Bağlan', style: TextStyle(fontSize: 20)),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class ConnectionScreen extends StatefulWidget {
  final String baseUrl;

  ConnectionScreen({required this.baseUrl});

  @override
  _ConnectionScreenState createState() => _ConnectionScreenState();
}

class _ConnectionScreenState extends State<ConnectionScreen> {
  bool isConnected = false;
  bool connectionFailed = false;
  String fullUrl = "";

  @override
  void initState() {
    super.initState();
    fullUrl = "http://${widget.baseUrl}:5000/get_number";
    checkConnection();
  }

  void checkConnection() async {
    final response = await http.get(Uri.parse(fullUrl));
    if (response.statusCode == 200) {
      final data = json.decode(response.body);
      final number = data['number'];

      setState(() {
        if (number == 42) {
          isConnected = true;
        } else {
          connectionFailed = true;
        }
      });
    } else {
      setState(() {
        connectionFailed = true;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Jetson Nano Park App'),
      ),
      body: Center(
        child: isConnected
            ? ElevatedButton(
                onPressed: () {
                  Navigator.push(
                    context,
                    MaterialPageRoute(builder: (context) => PlateInputScreen(baseUrl: widget.baseUrl)),
                  );
                },
                child: const Text('Jetson Nano Bağlandı, Devam Et', style: TextStyle(fontSize: 20)),
              )
            : connectionFailed
                ? const Text('Bağlantı başarısız', style: TextStyle(fontSize: 20))
                : const Text('Bağlantı kontrol ediliyor...', style: TextStyle(fontSize: 20)),
      ),
    );
  }
}

class PlateInputScreen extends StatelessWidget {
  final String baseUrl;
  final TextEditingController plateController = TextEditingController();
  final _formKey = GlobalKey<FormState>();

  PlateInputScreen({required this.baseUrl});

  void sendPlate(String plateNumber) async {
    final url = "http://$baseUrl:5000/plate";
    final response = await http.post(
      Uri.parse(url),
      headers: {'Content-Type': 'application/json'},
      body: json.encode({'plate': plateNumber}),
    );

    if (response.statusCode == 200) {
      print('Plaka başarıyla gönderildi');
    } else {
      print('Plaka gönderilemedi');
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Plaka Girişi'),
      ),
      body: Center(
        child: Form(
          key: _formKey,
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Padding(
                padding: const EdgeInsets.all(8.0),
                child: TextFormField(
                  controller: plateController,
                  decoration: const InputDecoration(
                    border: OutlineInputBorder(),
                    labelText: 'Plakayı Girin (0-9)',
                  ),
                  style: TextStyle(fontSize: 20),
                  keyboardType: TextInputType.number,
                  validator: (value) {
                    if (value == null || value.isEmpty) {
                      return 'Lütfen Plaka Numarası Giriniz';
                    }
                    return null;
                  },
                ),
              ),
              ElevatedButton(
                onPressed: () {
                  if (_formKey.currentState?.validate() ?? false) {
                    sendPlate(plateController.text);
                    Navigator.push(
                      context,
                      MaterialPageRoute(
                        builder: (context) => ParkingScreen(
                          baseUrl: baseUrl,
                          plateNumber: plateController.text,
                        ),
                      ),
                    );
                  }
                },
                child: const Text('Plakayı Gönder', style: TextStyle(fontSize: 20)),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class ParkingScreen extends StatefulWidget {
  final String baseUrl;
  final String plateNumber;

  ParkingScreen({required this.baseUrl, required this.plateNumber});

  @override
  _ParkingScreenState createState() => _ParkingScreenState();
}

class _ParkingScreenState extends State<ParkingScreen> {
  int score = 0;
  bool parkingCompleted = false;
  Timer? timer;

  @override
  void initState() {
    super.initState();
    fetchScore();
    checkParkingStatusPeriodically();
  }

  @override
  void dispose() {
    timer?.cancel(); // Timer'ı iptal et
    super.dispose();
  }

  void fetchScore() async {
    final url = "http://${widget.baseUrl}:5000/score";
    final response = await http.get(Uri.parse(url));
    if (response.statusCode == 200) {
      final data = json.decode(response.body);
      setState(() {
        score = data['score'];
      });
    } else {
      print('Skor alınamadı');
    }
  }

  void checkParkingStatus() async {
    final url = "http://${widget.baseUrl}:5000/get_park_complete";
    final response = await http.get(Uri.parse(url));
    if (response.statusCode == 200) {
      final data = json.decode(response.body);
      setState(() {
        parkingCompleted = data['park_status'] == "complete";
      });
    } else {
      print('Park durumu alınamadı');
    }
  }

  void checkParkingStatusPeriodically() {
    timer = Timer.periodic(Duration(seconds: 2), (timer) {
      checkParkingStatus();
      if (parkingCompleted) {
        timer.cancel();
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Park İşlemi'),
      ),
      body: Center(
        child: parkingCompleted
            ? Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  const Text('Park İşlemi Tamamlandı', style: TextStyle(fontSize: 20)),
                  ElevatedButton(
                    onPressed: () {
                      Navigator.pop(context);
                    },
                    child: const Text('Plaka Girmeye Dön', style: TextStyle(fontSize: 20)),
                  ),
                ],
              )
            : Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  const Text('Park işlemi devam ediyor...', style: TextStyle(fontSize: 20)),
                  Text('Puan: $score', style: const TextStyle(fontSize: 30)),
                ],
              ),
      ),
    );
  }
}
