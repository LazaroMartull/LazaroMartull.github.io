# mqtt_client.py
import paho.mqtt.client as mqtt
import ssl
import json
from datetime import datetime
import os
import tempfile
from queue import Queue

class SmartHomeMQTTClient:
    """Handles MQTT connection and communication using a thread-safe queue."""

    def __init__(self, config):
        self.client = None
        self.config = config
        self.is_connected = False
        self._temp_files = []
        self.update_queue = Queue()

    def _log(self, message):
        """Puts a log message into the thread-safe queue."""
        self.update_queue.put(('log', message))

    def _on_connect(self, client, userdata, flags, rc, properties=None):
        """Callback for when the client connects."""
        if rc == 0:
            self.is_connected = True
            self._log(f"✅ Successfully connected to AWS IoT broker.")
            client.subscribe(self.config['topic_subscribe'])
            self._log(f"✅ Subscribed to topic: {self.config['topic_subscribe']}")
            self.update_queue.put(('status', True))
        else:
            self.is_connected = False
            self._log(f"❌ Failed to connect, return code {rc}")
            self.update_queue.put(('status', False))

    def _on_message(self, client, userdata, msg):
        """Callback for when a message is received."""
        try:
            payload = msg.payload.decode('utf-8')
            self._log(f"📨 Message received on topic: {msg.topic}")
            self.update_queue.put(('raw_payload', f"[{datetime.now().strftime('%H:%M:%S')}] {payload}"))

            data = json.loads(payload)
            self.update_queue.put(('data', data))
            self._log("✅ Parsed and queued new data.")
        except json.JSONDecodeError:
            self._log(f"❌ Error decoding JSON from payload: {msg.payload.decode('utf-8')}")
        except Exception as e:
            self._log(f"❌ Error processing message: {e}")

    def _on_disconnect(self, client, userdata, rc, properties=None):
        """Callback for when the client disconnects."""
        self.is_connected = False
        self._log(f"🔌 Client disconnected with result code: {rc}")
        self.update_queue.put(('status', False))
        self._cleanup_certs()

    def _create_temp_cert_file(self, content):
        """Creates a temporary file for a certificate."""
        fd, path = tempfile.mkstemp(suffix=".pem")
        with os.fdopen(fd, 'w') as tmp:
            tmp.write(content)
        self._temp_files.append(path)
        return path

    def _cleanup_certs(self):
        """Removes temporary certificate files."""
        for path in self._temp_files:
            try: os.remove(path)
            except OSError: pass
        self._temp_files = []

    def connect(self):
        """Sets up and connects the MQTT client."""
        try:
            self._log("🔄 Initializing MQTT connection...")
            ca_path = self._create_temp_cert_file(self.config['ca_cert'])
            cert_path = self._create_temp_cert_file(self.config['device_cert'])
            key_path = self._create_temp_cert_file(self.config['private_key'])

            self.client = mqtt.Client(client_id=f"dashboard-{int(datetime.now().timestamp())}", protocol=mqtt.MQTTv5)
            self.client.on_connect = self._on_connect
            self.client.on_message = self._on_message
            self.client.on_disconnect = self._on_disconnect

            self.client.tls_set(ca_certs=ca_path, certfile=cert_path, keyfile=key_path, cert_reqs=ssl.CERT_REQUIRED, tls_version=ssl.PROTOCOL_TLS)
            self.client.connect(self.config['broker'], self.config['port'], 60)
            self.client.loop_start()
        except Exception as e:
            self._log(f"❌ Exception during connection setup: {e}")
            self.disconnect()

    def get_updates(self):
        """Retrieves all pending updates from the queue."""
        updates = []
        while not self.update_queue.empty():
            updates.append(self.update_queue.get())
        return updates

    def publish(self, command):
        """Publishes a command to the ESP32."""
        if not self.is_connected:
            self._log("❌ Publish failed: Client not connected.")
            return

        try:
            payload = json.dumps(command)
            result = self.client.publish(self.config['topic_publish'], payload)
            result.wait_for_publish(timeout=5)
            if result.is_published():
                self._log(f"📤 Command published: {payload}")
            else:
                self._log("❌ Publish confirmation timed out.")
        except Exception as e:
            self._log(f"❌ Exception during publish: {e}")

    def disconnect(self):
        """Stops the loop and disconnects the client."""
        if self.client:
            self.client.loop_stop()
            self.client.disconnect()
        self._cleanup_certs()
        self._log("⚪ MQTT client disconnected and certificates cleaned up.")
