"""
pyhik.isapi
~~~~~~~~~~~
ISAPI client for Hikvision devices.
Provides comprehensive access to Hikvision ISAPI endpoints.

Copyright (c) 2016-2026 John Mihalic <https://github.com/mezz64>
Licensed under the MIT license.
"""

from dataclasses import dataclass, field
import datetime as dt
from enum import Enum
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid
import xml.etree.ElementTree as ET

import requests
from requests.auth import HTTPBasicAuth, HTTPDigestAuth

try:
    import xmltodict
except ImportError:
    xmltodict = None

_LOGGER = logging.getLogger(__name__)

# ISAPI Endpoints
ENDPOINT_DEVICE_INFO = "/ISAPI/System/deviceInfo"
ENDPOINT_CAPABILITIES = "/ISAPI/System/capabilities"
ENDPOINT_STORAGE = "/ISAPI/ContentMgmt/Storage"
ENDPOINT_STREAMING_CHANNELS = "/ISAPI/Streaming/channels"
ENDPOINT_INPUT_PROXY_CHANNELS = "/ISAPI/ContentMgmt/InputProxy/channels"
ENDPOINT_IO_INPUTS = "/ISAPI/System/IO/inputs"
ENDPOINT_IO_OUTPUTS = "/ISAPI/System/IO/outputs"
ENDPOINT_EVENT_NOTIFICATION = "/ISAPI/Event/notification/httpHosts"
ENDPOINT_HOLIDAYS = "/ISAPI/System/Holidays"
ENDPOINT_REBOOT = "/ISAPI/System/reboot"
ENDPOINT_EVENT_TRIGGERS = "/ISAPI/Event/triggers"
ENDPOINT_SMART_CAPABILITIES = "/ISAPI/Smart/capabilities"
ENDPOINT_CONTENTMGMT_SEARCH = "/ISAPI/ContentMgmt/search"

# Recording search timeout (larger than default, searches can be slow)
RECORDING_SEARCH_TIMEOUT = 30

# Event detection endpoints
EVENT_ENDPOINTS: Dict[str, str] = {
    "motionDetection": "/ISAPI/System/Video/inputs/channels/{channel}/motionDetection",
    "lineDetection": "/ISAPI/Smart/LineDetection/{channel}",
    "fieldDetection": "/ISAPI/Smart/FieldDetection/{channel}",
    "regionEntrance": "/ISAPI/Smart/RegionEntrance/{channel}",
    "regionExiting": "/ISAPI/Smart/RegionExiting/{channel}",
    "tamperDetection": "/ISAPI/System/Video/inputs/channels/{channel}/tamperDetection",
    "sceneChangeDetection": "/ISAPI/Smart/SceneChangeDetection/{channel}",
    "PIR": "/ISAPI/WLAlarm/PIR",
    "faceDetection": "/ISAPI/Smart/FaceDetect/{channel}",
}

# Request timeout
REQUEST_TIMEOUT = 20


class HTTPMethod(Enum):
    """HTTP methods."""

    GET = "GET"
    PUT = "PUT"
    POST = "POST"
    DELETE = "DELETE"


class ISAPIError(Exception):
    """Base ISAPI error."""


class ISAPIConnectionError(ISAPIError):
    """Connection error."""


class ISAPIAuthError(ISAPIError):
    """Authentication error."""


class ISAPINotFoundError(ISAPIError):
    """Resource not found error."""


@dataclass
class StorageDevice:
    """Storage device information."""

    id: str
    name: str
    status: str
    type: str
    capacity: Optional[int] = None
    free_space: Optional[int] = None
    ip_address: Optional[str] = None


@dataclass
class AlarmServerInfo:
    """Alarm server configuration."""

    protocol: str = ""
    address: str = ""
    port: int = 0
    path: str = ""


@dataclass
class StreamInfo:
    """Camera stream information."""

    id: str
    channel_id: int
    type_id: int
    name: str
    enabled: bool


@dataclass
class CameraInfo:
    """Camera information."""

    id: int
    name: str
    model: Optional[str] = None
    serial_number: Optional[str] = None
    input_port: Optional[int] = None
    streams: List["StreamInfo"] = field(default_factory=list)


@dataclass
class OutputPort:
    """Output port information."""

    id: str
    name: str


@dataclass
class InputPort:
    """Input port information."""

    id: str
    name: str


@dataclass
class EventState:
    """Event detection state."""

    id: str
    channel: int
    type: str
    enabled: bool


@dataclass
class DeviceCapabilities:
    """Device capabilities."""

    support_holiday_mode: bool = False
    support_alarm_server: bool = False
    support_io_outputs: bool = False
    support_io_inputs: bool = False
    support_storage: bool = False
    num_io_outputs: int = 0
    num_io_inputs: int = 0


@dataclass
class Recording:
    """Video recording segment."""

    source_id: str
    track_id: int
    start_time: dt.datetime
    end_time: dt.datetime
    content_type: str = "video"
    playback_uri: str = ""


@dataclass
class RecordingDay:
    """Day with recordings available."""

    date: dt.datetime
    has_recordings: bool = True


class ISAPIClient:
    """Client for Hikvision ISAPI.

    Provides synchronous access to Hikvision device ISAPI endpoints.
    """

    def __init__(
        self,
        host: str,
        port: int = 80,
        username: str = "",
        password: str = "",
        ssl: bool = False,
        verify_ssl: bool = True,
        rtsp_port: int = 554,
    ) -> None:
        """Initialize the ISAPI client.

        Args:
            host: Device hostname or IP address.
            port: HTTP port (default 80).
            username: Authentication username.
            password: Authentication password.
            ssl: Use HTTPS if True.
            verify_ssl: Verify SSL certificates.
            rtsp_port: RTSP port for streaming (default 554).
        """
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.ssl = ssl
        self.verify_ssl = verify_ssl
        self.rtsp_port = rtsp_port

        protocol = "https" if ssl else "http"
        self.base_url = f"{protocol}://{host}:{port}"

        self._session = requests.Session()
        self._session.verify = verify_ssl
        self._auth: Optional[Union[HTTPBasicAuth, HTTPDigestAuth]] = None
        self._device_info: Dict[str, Any] = {}
        self._capabilities: Optional[DeviceCapabilities] = None

    def _detect_auth_method(self) -> None:
        """Detect the authentication method (Basic or Digest)."""
        if self._auth is not None:
            return

        url = f"{self.base_url}{ENDPOINT_DEVICE_INFO}"

        # Try digest auth first (more common for Hikvision)
        try:
            digest_auth = HTTPDigestAuth(self.username, self.password)
            response = self._session.get(url, auth=digest_auth, timeout=REQUEST_TIMEOUT)
            if response.status_code == 200:
                self._auth = digest_auth
                return
        except Exception:
            pass

        # Fall back to basic auth
        self._auth = HTTPBasicAuth(self.username, self.password)

    def _parse_xml(self, text: str) -> Dict[str, Any]:
        """Parse XML response to dictionary."""
        if xmltodict is None:
            raise ISAPIError(
                "xmltodict is required for ISAPI client. "
                "Install it with: pip install xmltodict"
            )
        try:
            return xmltodict.parse(text)
        except Exception:
            return {"raw": text}

    def _unparse_xml(self, data: Dict[str, Any]) -> str:
        """Convert dictionary to XML string."""
        if xmltodict is None:
            raise ISAPIError(
                "xmltodict is required for ISAPI client. "
                "Install it with: pip install xmltodict"
            )
        return xmltodict.unparse(data)

    def request(
        self,
        method: HTTPMethod,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Union[Dict[str, Any], bytes]:
        """Make an ISAPI request.

        Args:
            method: HTTP method to use.
            endpoint: ISAPI endpoint path.
            data: Optional data to send (will be converted to XML).
            params: Optional query parameters.

        Returns:
            Parsed XML response as dictionary, or raw bytes for binary content.

        Raises:
            ISAPIConnectionError: Connection failed.
            ISAPIAuthError: Authentication failed.
            ISAPINotFoundError: Endpoint not found.
            ISAPIError: Other request errors.
        """
        self._detect_auth_method()

        url = f"{self.base_url}{endpoint}"

        try:
            if method == HTTPMethod.GET:
                response = self._session.get(
                    url, auth=self._auth, params=params, timeout=REQUEST_TIMEOUT
                )
            elif method == HTTPMethod.PUT:
                xml_data = self._unparse_xml(data) if data else None
                response = self._session.put(
                    url,
                    auth=self._auth,
                    data=xml_data,
                    headers={"Content-Type": "application/xml"},
                    timeout=REQUEST_TIMEOUT,
                )
            elif method == HTTPMethod.POST:
                xml_data = self._unparse_xml(data) if data else None
                response = self._session.post(
                    url,
                    auth=self._auth,
                    data=xml_data,
                    headers={"Content-Type": "application/xml"},
                    timeout=REQUEST_TIMEOUT,
                )
            elif method == HTTPMethod.DELETE:
                response = self._session.delete(
                    url, auth=self._auth, timeout=REQUEST_TIMEOUT
                )
            else:
                response = self._session.request(
                    method.value, url, auth=self._auth, timeout=REQUEST_TIMEOUT
                )

        except requests.exceptions.ConnectionError as err:
            raise ISAPIConnectionError(f"Cannot connect to {self.host}") from err
        except requests.exceptions.Timeout as err:
            raise ISAPIConnectionError(f"Timeout connecting to {self.host}") from err
        except requests.exceptions.RequestException as err:
            raise ISAPIConnectionError(f"Request failed: {err}") from err

        if response.status_code == 401:
            raise ISAPIAuthError("Invalid credentials")
        if response.status_code == 403:
            raise ISAPIAuthError("Insufficient permissions")
        if response.status_code == 404:
            raise ISAPINotFoundError(f"Endpoint not found: {endpoint}")
        if response.status_code >= 400:
            raise ISAPIError(f"Request failed with status {response.status_code}")

        content_type = response.headers.get("content-type", "")
        if "image" in content_type or "octet-stream" in content_type:
            return response.content

        return self._parse_xml(response.text)

    def get_device_info(self) -> Dict[str, Any]:
        """Get device information."""
        if not self._device_info:
            response = self.request(HTTPMethod.GET, ENDPOINT_DEVICE_INFO)
            self._device_info = response.get("DeviceInfo", {})
        return self._device_info

    def get_device_serial(self) -> str:
        """Get device serial number."""
        info = self.get_device_info()
        return info.get("serialNumber", "")

    def get_device_name(self) -> str:
        """Get device name."""
        info = self.get_device_info()
        return info.get("deviceName", "")

    def get_device_model(self) -> str:
        """Get device model."""
        info = self.get_device_info()
        return info.get("model", "")

    def get_device_type(self) -> str:
        """Get device type (e.g., NVR, DVR, IPCamera)."""
        info = self.get_device_info()
        return info.get("deviceType", "Camera")

    def get_firmware_version(self) -> str:
        """Get device firmware version."""
        info = self.get_device_info()
        return info.get("firmwareVersion", "")

    def get_capabilities(self) -> DeviceCapabilities:
        """Get device capabilities."""
        if self._capabilities is not None:
            return self._capabilities

        capabilities = DeviceCapabilities()

        # Check for holiday mode support
        try:
            self.request(HTTPMethod.GET, ENDPOINT_HOLIDAYS)
            capabilities.support_holiday_mode = True
        except (ISAPINotFoundError, ISAPIError):
            pass

        # Check for alarm server support
        try:
            self.request(HTTPMethod.GET, ENDPOINT_EVENT_NOTIFICATION)
            capabilities.support_alarm_server = True
        except (ISAPINotFoundError, ISAPIError):
            pass

        # Check for IO outputs
        try:
            response = self.request(HTTPMethod.GET, ENDPOINT_IO_OUTPUTS)
            outputs = response.get("IOOutputPortList", {}).get("IOOutputPort", [])
            if isinstance(outputs, dict):
                outputs = [outputs]
            capabilities.support_io_outputs = len(outputs) > 0
            capabilities.num_io_outputs = len(outputs)
        except (ISAPINotFoundError, ISAPIError):
            pass

        # Check for IO inputs
        try:
            response = self.request(HTTPMethod.GET, ENDPOINT_IO_INPUTS)
            inputs = response.get("IOInputPortList", {}).get("IOInputPort", [])
            if isinstance(inputs, dict):
                inputs = [inputs]
            capabilities.support_io_inputs = len(inputs) > 0
            capabilities.num_io_inputs = len(inputs)
        except (ISAPINotFoundError, ISAPIError):
            pass

        # Check for storage
        try:
            self.request(HTTPMethod.GET, ENDPOINT_STORAGE)
            capabilities.support_storage = True
        except (ISAPINotFoundError, ISAPIError):
            pass

        self._capabilities = capabilities
        return capabilities

    def get_storage_devices(self) -> List[StorageDevice]:
        """Get storage device information."""
        try:
            response = self.request(HTTPMethod.GET, ENDPOINT_STORAGE)
        except ISAPINotFoundError:
            return []

        devices = []
        storage_list = response.get("storage") or {}

        # Handle HDD list
        hdd_list = (storage_list.get("hddList") or {}).get("hdd", [])
        if isinstance(hdd_list, dict):
            hdd_list = [hdd_list]

        for hdd in hdd_list:
            devices.append(
                StorageDevice(
                    id=hdd.get("id", ""),
                    name=hdd.get("hddName", f"HDD {hdd.get('id', '')}"),
                    status=hdd.get("status", "unknown"),
                    type="HDD",
                    capacity=self._parse_capacity(hdd.get("capacity")),
                    free_space=self._parse_capacity(hdd.get("freeSpace")),
                )
            )

        # Handle NAS list
        nas_list = (storage_list.get("nasList") or {}).get("nas", [])
        if isinstance(nas_list, dict):
            nas_list = [nas_list]

        for nas in nas_list:
            devices.append(
                StorageDevice(
                    id=nas.get("id", ""),
                    name=nas.get("nasName", f"NAS {nas.get('id', '')}"),
                    status=nas.get("status", "unknown"),
                    type="NAS",
                    capacity=self._parse_capacity(nas.get("capacity")),
                    free_space=self._parse_capacity(nas.get("freeSpace")),
                    ip_address=nas.get("ipAddress"),
                )
            )

        return devices

    def _parse_capacity(self, value: Optional[str]) -> Optional[int]:
        """Parse capacity value to bytes.

        Hikvision returns storage values in MB, convert to bytes.
        """
        if value is None:
            return None
        try:
            return int(value) * 1024 * 1024
        except (ValueError, TypeError):
            return None

    def get_alarm_server_info(self) -> AlarmServerInfo:
        """Get alarm server configuration."""
        try:
            response = self.request(HTTPMethod.GET, ENDPOINT_EVENT_NOTIFICATION)
        except ISAPINotFoundError:
            return AlarmServerInfo()

        hosts = response.get("HttpHostNotificationList", {}).get(
            "HttpHostNotification", []
        )
        if isinstance(hosts, dict):
            hosts = [hosts]

        if not hosts:
            return AlarmServerInfo()

        host = hosts[0]
        return AlarmServerInfo(
            protocol=host.get("protocolType", ""),
            address=host.get("ipAddress", host.get("hostName", "")),
            port=int(host.get("portNo", 0)),
            path=host.get("url", ""),
        )

    def get_streaming_channels(self) -> List[StreamInfo]:
        """Get streaming channel information."""
        # Try standard streaming channels endpoint first
        try:
            response = self.request(HTTPMethod.GET, ENDPOINT_STREAMING_CHANNELS)
            channels = response.get("StreamingChannelList", {}).get(
                "StreamingChannel", []
            )
            if isinstance(channels, dict):
                channels = [channels]

            streams = []
            for channel in channels:
                channel_id = channel.get("id", "")
                try:
                    full_id = int(channel_id)
                    cam_id = full_id // 100
                    stream_type = full_id % 100
                except (ValueError, TypeError):
                    continue

                streams.append(
                    StreamInfo(
                        id=channel_id,
                        channel_id=cam_id,
                        type_id=stream_type,
                        name=channel.get("channelName", f"Channel {cam_id}"),
                        enabled=channel.get("enabled", "true").lower() == "true",
                    )
                )

            if streams:
                return streams
        except (ISAPINotFoundError, ISAPIError):
            pass

        # Try NVR input proxy channels endpoint
        try:
            response = self.request(HTTPMethod.GET, ENDPOINT_INPUT_PROXY_CHANNELS)
            channels = response.get("InputProxyChannelList", {}).get(
                "InputProxyChannel", []
            )
            if isinstance(channels, dict):
                channels = [channels]

            streams = []
            for channel in channels:
                try:
                    channel_id = int(channel.get("id", 0))
                except (ValueError, TypeError):
                    continue

                channel_name = channel.get("name", f"Channel {channel_id}")
                # Create main stream (type 1) for each NVR channel
                streams.append(
                    StreamInfo(
                        id=f"{channel_id}01",
                        channel_id=channel_id,
                        type_id=1,
                        name=channel_name,
                        enabled=True,
                    )
                )
                # Create sub stream (type 2) for each NVR channel
                streams.append(
                    StreamInfo(
                        id=f"{channel_id}02",
                        channel_id=channel_id,
                        type_id=2,
                        name=channel_name,
                        enabled=True,
                    )
                )

            return streams
        except (ISAPINotFoundError, ISAPIError):
            return []

    def get_input_proxy_channels(self) -> List[CameraInfo]:
        """Get NVR input proxy channels (connected cameras).

        Returns:
            List of CameraInfo (without streams populated).
        """
        response = self.request(HTTPMethod.GET, ENDPOINT_INPUT_PROXY_CHANNELS)
        channels = response.get("InputProxyChannelList", {}).get(
            "InputProxyChannel", []
        )
        if isinstance(channels, dict):
            channels = [channels]

        cameras: List[CameraInfo] = []
        for ch in channels:
            try:
                ch_id = int(ch.get("id", 0))
            except (ValueError, TypeError):
                continue
            cameras.append(
                CameraInfo(
                    id=ch_id,
                    name=ch.get("name", f"Channel {ch_id}"),
                    model=ch.get("model"),
                    serial_number=ch.get("serialNumber"),
                    input_port=(
                        int(ch.get("inputPort"))
                        if ch.get("inputPort") is not None
                        else None
                    ),
                )
            )
        return cameras

    def find_camera(self, name: str) -> Optional[CameraInfo]:
        """Find a camera by name (supports Chinese, fuzzy match).

        Args:
            name: Camera name or partial name to search.

        Returns:
            CameraInfo if found, None otherwise.
        """
        cameras = self.get_input_proxy_channels()
        for cam in cameras:
            if name in cam.name:
                return cam
        return None

    def download_recordings(
        self,
        camera: str,
        start_date: dt.datetime,
        end_date: dt.datetime,
        output_dir: str = ".",
    ) -> List[str]:
        """Download all recordings for a camera in a date range.

        Args:
            camera: Camera name (e.g., "卧室") or channel number.
            start_date: Start date.
            end_date: End date.
            output_dir: Directory to save files (default: current dir).

        Returns:
            List of downloaded file paths.
        """
        # Resolve camera
        cam: Optional[CameraInfo] = None
        if isinstance(camera, str):
            cam = self.find_camera(camera)
            if cam is None:
                try:
                    channel = int(camera)
                except ValueError:
                    raise ISAPIError(f"Camera not found: {camera}")
            else:
                channel = cam.id
        else:
            channel = camera

        # Search all recordings in date range
        recordings = self.search_recordings(
            channel=channel,
            start_time=start_date,
            end_time=end_date,
            max_results=1000,
        )

        if not recordings:
            _LOGGER.info("No recordings found for channel %s", channel)
            return []

        # Download each recording
        import os

        downloaded: List[str] = []
        cam_name = cam.name if cam else f"ch{channel}"
        safe_name = cam_name.replace("/", "_").replace(" ", "_")

        for i, rec in enumerate(recordings):
            ts = rec.start_time.strftime("%Y%m%d_%H%M%S")
            filename = f"{safe_name}_{ts}_{i}.mp4"
            filepath = os.path.join(output_dir, filename)

            _LOGGER.info(
                "Downloading %s ~ %s -> %s",
                rec.start_time, rec.end_time, filename,
            )
            self.download_recording(rec.playback_uri, filepath)
            downloaded.append(filepath)

        return downloaded

    def get_cameras(self) -> List[CameraInfo]:
        """Get camera information with streams."""
        streams = self.get_streaming_channels()

        cameras_dict: Dict[int, CameraInfo] = {}
        for stream in streams:
            if stream.channel_id not in cameras_dict:
                cameras_dict[stream.channel_id] = CameraInfo(
                    id=stream.channel_id,
                    name=stream.name,
                    streams=[],
                )
            cameras_dict[stream.channel_id].streams.append(stream)

        return list(cameras_dict.values())

    def get_output_ports(self) -> List[OutputPort]:
        """Get output port information."""
        try:
            response = self.request(HTTPMethod.GET, ENDPOINT_IO_OUTPUTS)
        except ISAPINotFoundError:
            return []

        outputs = response.get("IOOutputPortList", {}).get("IOOutputPort", [])
        if isinstance(outputs, dict):
            outputs = [outputs]

        return [
            OutputPort(
                id=output.get("id", ""),
                name=output.get("outputName", f"Output {output.get('id', '')}"),
            )
            for output in outputs
        ]

    def get_input_ports(self) -> List[InputPort]:
        """Get input port information."""
        try:
            response = self.request(HTTPMethod.GET, ENDPOINT_IO_INPUTS)
        except ISAPINotFoundError:
            return []

        inputs = response.get("IOInputPortList", {}).get("IOInputPort", [])
        if isinstance(inputs, dict):
            inputs = [inputs]

        return [
            InputPort(
                id=inp.get("id", ""),
                name=inp.get("inputName", f"Input {inp.get('id', '')}"),
            )
            for inp in inputs
        ]

    def get_output_state(self, output_id: str) -> bool:
        """Get output port state."""
        try:
            response = self.request(
                HTTPMethod.GET, f"{ENDPOINT_IO_OUTPUTS}/{output_id}/status"
            )
            status = response.get("IOPortStatus", {})
            return status.get("ioState", "inactive").lower() == "active"
        except ISAPIError:
            return False

    def set_output_state(self, output_id: str, state: bool) -> None:
        """Set output port state."""
        data = {
            "IOPortData": {
                "@version": "2.0",
                "@xmlns": "http://www.isapi.org/ver20/XMLSchema",
                "outputState": "high" if state else "low",
            }
        }
        self.request(
            HTTPMethod.PUT, f"{ENDPOINT_IO_OUTPUTS}/{output_id}/trigger", data=data
        )

    def get_holiday_mode_enabled(self) -> bool:
        """Get holiday mode status."""
        try:
            response = self.request(HTTPMethod.GET, ENDPOINT_HOLIDAYS)
            holidays = response.get("HolidayList", {}).get("holiday", [])
            if isinstance(holidays, dict):
                holidays = [holidays]
            for h in holidays:
                enabled = h.get("enabled", "false")
                if isinstance(enabled, dict):
                    enabled = enabled.get("#text", "false")
                if str(enabled).lower() == "true":
                    return True
            return False
        except ISAPIError:
            return False

    def set_holiday_mode_enabled(self, enabled: bool) -> None:
        """Set holiday mode status."""
        try:
            response = self.request(HTTPMethod.GET, ENDPOINT_HOLIDAYS)
        except ISAPIError:
            return

        holidays = response.get("HolidayList", {}).get("holiday", [])
        if isinstance(holidays, dict):
            holidays = [holidays]

        if not holidays:
            return

        holidays[0]["enabled"] = "true" if enabled else "false"
        data = {"HolidayList": {"holiday": holidays}}
        self.request(HTTPMethod.PUT, ENDPOINT_HOLIDAYS, data=data)

    def get_event_states(self) -> List[EventState]:
        """Get event detection states for all channels."""
        states = []
        cameras = self.get_cameras()

        for camera in cameras:
            channel = camera.id
            for event_type, endpoint_template in EVENT_ENDPOINTS.items():
                endpoint = endpoint_template.format(channel=channel)
                try:
                    response = self.request(HTTPMethod.GET, endpoint)
                    for value in response.values():
                        if isinstance(value, dict) and "enabled" in value:
                            enabled = value.get("enabled", "false").lower() == "true"
                            states.append(
                                EventState(
                                    id=f"{event_type}_{channel}",
                                    channel=channel,
                                    type=event_type,
                                    enabled=enabled,
                                )
                            )
                            break
                except ISAPIError:
                    continue

        return states

    def set_event_enabled(
        self, event_type: str, channel: int, enabled: bool
    ) -> None:
        """Set event detection enabled state."""
        endpoint_template = EVENT_ENDPOINTS.get(event_type)
        if not endpoint_template:
            raise ISAPIError(f"Unknown event type: {event_type}")

        endpoint = endpoint_template.format(channel=channel)

        try:
            response = self.request(HTTPMethod.GET, endpoint)
        except ISAPIError as err:
            raise ISAPIError(f"Cannot get event config: {err}") from err

        for value in response.values():
            if isinstance(value, dict) and "enabled" in value:
                value["enabled"] = "true" if enabled else "false"
                break

        self.request(HTTPMethod.PUT, endpoint, data=response)

    def get_snapshot(
        self,
        channel: int = 1,
        stream_type: int = 1,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> bytes:
        """Get camera snapshot.

        Args:
            channel: Camera channel number (default 1).
            stream_type: Stream type (1=main, 2=sub, default 1).
            width: Optional image width.
            height: Optional image height.

        Returns:
            Image data as bytes.
        """
        stream_id = channel * 100 + stream_type
        endpoint = f"{ENDPOINT_STREAMING_CHANNELS}/{stream_id}/picture"

        params = {}
        if width:
            params["width"] = width
        if height:
            params["height"] = height

        result = self.request(HTTPMethod.GET, endpoint, params=params or None)
        if isinstance(result, bytes):
            return result
        raise ISAPIError("Failed to get snapshot")

    def get_rtsp_url(
        self,
        channel: int = 1,
        stream_type: int = 1,
        include_credentials: bool = True,
    ) -> str:
        """Get RTSP URL for a channel.

        Args:
            channel: Camera channel number (default 1).
            stream_type: Stream type (1=main, 2=sub, default 1).
            include_credentials: Include username/password in URL.

        Returns:
            RTSP URL string.
        """
        stream_id = channel * 100 + stream_type
        protocol = "rtsps" if self.ssl else "rtsp"

        if include_credentials:
            return (
                f"{protocol}://{self.username}:{self.password}@"
                f"{self.host}:{self.rtsp_port}/Streaming/Channels/{stream_id}"
            )
        return (
            f"{protocol}://{self.host}:{self.rtsp_port}"
            f"/Streaming/Channels/{stream_id}"
        )

    def reboot(self) -> None:
        """Reboot the device."""
        self.request(HTTPMethod.PUT, ENDPOINT_REBOOT)

    def custom_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Make a custom ISAPI request.

        Args:
            method: HTTP method (GET, PUT, POST, DELETE).
            endpoint: ISAPI endpoint path.
            data: Optional XML data string.

        Returns:
            Parsed response as dictionary.
        """
        http_method = HTTPMethod(method.upper())

        if data:
            try:
                parsed_data = self._parse_xml(data)
            except Exception as err:
                raise ISAPIError(f"Invalid XML data: {err}") from err
        else:
            parsed_data = None

        result = self.request(http_method, endpoint, data=parsed_data)
        if isinstance(result, bytes):
            return {"raw_bytes": True, "length": len(result)}
        return result

    # --- Recording search / download -----------------------------------

    @staticmethod
    def _track_id(channel: int, stream_type: int = 1) -> int:
        """Convert channel number to ISAPI track ID."""
        return channel * 100 + stream_type

    def get_recording_days(
        self,
        channel: int = 1,
        start_date: Optional[dt.datetime] = None,
        end_date: Optional[dt.datetime] = None,
    ) -> List[RecordingDay]:
        """Get days with recordings for a channel.

        Args:
            channel: Camera channel number (default 1).
            start_date: Start date (default: 30 days ago).
            end_date: End date (default: today).

        Returns:
            List of RecordingDay sorted by date descending.
        """
        if end_date is None:
            end_date = dt.datetime.now()
        if start_date is None:
            start_date = end_date - dt.timedelta(days=30)

        days: Dict[str, RecordingDay] = {}
        window = dt.timedelta(days=1)
        current = start_date

        while current < end_date:
            window_end = min(current + window, end_date)
            try:
                results = self.search_recordings(
                    channel=channel,
                    start_time=current,
                    end_time=window_end,
                    max_results=500,
                )
                for rec in results:
                    key = rec.start_time.strftime("%Y-%m-%d")
                    if key not in days:
                        days[key] = RecordingDay(
                            date=rec.start_time.replace(
                                hour=0, minute=0, second=0, microsecond=0
                            ),
                            has_recordings=True,
                        )
            except ISAPIError:
                pass

            current = window_end

        return sorted(days.values(), key=lambda d: d.date, reverse=True)

    def search_recordings(
        self,
        channel: int = 1,
        start_time: Optional[dt.datetime] = None,
        end_time: Optional[dt.datetime] = None,
        max_results: int = 100,
    ) -> List[Recording]:
        """Search recordings by channel and time range.

        Args:
            channel: Camera channel number (default 1).
            start_time: Start time (default: 1 hour ago).
            end_time: End time (default: now).
            max_results: Maximum results to return.

        Returns:
            List of Recording sorted by start_time descending.
        """
        if end_time is None:
            end_time = dt.datetime.now()
        if start_time is None:
            start_time = end_time - dt.timedelta(hours=1)

        track_id = self._track_id(channel)

        search_xml = (
            '<?xml version="1.0" encoding="utf-8"?>'
            "<CMSearchDescription>"
            "<searchID>{search_id}</searchID>"
            "<trackIDList><trackID>{track_id}</trackID></trackIDList>"
            "<timeSpanList>"
            "<timeSpan>"
            "<startTime>{start}Z</startTime>"
            "<endTime>{end}Z</endTime>"
            "</timeSpan>"
            "</timeSpanList>"
            "<maxResults>{max_results}</maxResults>"
            "<searchResultPosition>0</searchResultPosition>"
            "<metadataList>"
            "<metadataDescriptor>//recordType.meta.std-cgi.com</metadataDescriptor>"
            "</metadataList>"
            "</CMSearchDescription>"
        ).format(
            search_id=str(uuid.uuid4()).upper(),
            track_id=track_id,
            start=start_time.strftime("%Y-%m-%dT%H:%M:%S"),
            end=end_time.strftime("%Y-%m-%dT%H:%M:%S"),
            max_results=max_results,
        )

        self._detect_auth_method()
        url = f"{self.base_url}{ENDPOINT_CONTENTMGMT_SEARCH}"

        try:
            response = self._session.post(
                url,
                auth=self._auth,
                data=search_xml,
                headers={"Content-Type": "application/xml"},
                timeout=RECORDING_SEARCH_TIMEOUT,
            )
        except requests.exceptions.RequestException as err:
            raise ISAPIConnectionError(f"Recording search failed: {err}") from err

        if response.status_code == 401:
            raise ISAPIAuthError("Invalid credentials")
        if response.status_code == 404:
            raise ISAPINotFoundError(
                f"Search endpoint not found: {ENDPOINT_CONTENTMGMT_SEARCH}"
            )
        if response.status_code >= 400:
            raise ISAPIError(
                f"Recording search failed with status {response.status_code}"
            )

        return self._parse_recording_results(response.text)

    def _parse_recording_results(self, xml_text: str) -> List[Recording]:
        """Parse CMSearchResult XML into Recording list."""
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError:
            return []

        recordings: List[Recording] = []

        for match in root.iter():
            if "searchMatchItem" not in match.tag:
                continue

            source_id = ""
            track_id_val = 0
            rec_start = None
            rec_end = None
            playback_uri = ""
            content_type = "video"

            for child in match:
                tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag

                if tag == "sourceID" and child.text:
                    source_id = child.text
                elif tag == "trackID" and child.text:
                    try:
                        track_id_val = int(child.text)
                    except ValueError:
                        pass
                elif tag == "timeSpan":
                    for tc in child:
                        ttag = tc.tag.split("}")[-1] if "}" in tc.tag else tc.tag
                        if ttag == "startTime" and tc.text:
                            rec_start = self._parse_iso(tc.text)
                        elif ttag == "endTime" and tc.text:
                            rec_end = self._parse_iso(tc.text)
                elif tag == "mediaSegmentDescriptor":
                    for mc in child:
                        mtag = mc.tag.split("}")[-1] if "}" in mc.tag else mc.tag
                        if mtag == "playbackURI" and mc.text:
                            playback_uri = mc.text
                        elif mtag == "contentType" and mc.text:
                            content_type = mc.text

            if rec_start is not None and rec_end is not None:
                recordings.append(
                    Recording(
                        source_id=source_id,
                        track_id=track_id_val,
                        start_time=rec_start,
                        end_time=rec_end,
                        content_type=content_type,
                        playback_uri=playback_uri,
                    )
                )

        return sorted(recordings, key=lambda r: r.start_time, reverse=True)

    @staticmethod
    def _parse_iso(text: str) -> Optional[dt.datetime]:
        """Parse ISO 8601 timestamp, with or without Z suffix."""
        try:
            return dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            try:
                return dt.datetime.fromisoformat(text.rstrip("Z"))
            except ValueError:
                return None

    def download_recording(
        self,
        playback_uri: str,
        output_path: str,
        chunk_size: int = 8192,
    ) -> None:
        """Download a recording to a local file.

        Args:
            playback_uri: The playback URI from a Recording.
            output_path: Local file path to save to.
            chunk_size: Stream chunk size in bytes.

        Raises:
            ISAPIError: Download failed.
        """
        url = playback_uri
        if url.startswith("/"):
            url = f"{self.base_url}{url}"

        self._detect_auth_method()

        try:
            response = self._session.get(
                url,
                auth=self._auth,
                stream=True,
                timeout=(10, 60),  # (connect, read)
            )
        except requests.exceptions.RequestException as err:
            raise ISAPIConnectionError(f"Download failed: {err}") from err

        if response.status_code == 401:
            raise ISAPIAuthError("Invalid credentials")
        if response.status_code == 404:
            raise ISAPINotFoundError(f"Recording not found: {playback_uri}")
        if response.status_code >= 400:
            raise ISAPIError(f"Download failed with status {response.status_code}")

        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)

    # --- context manager ---

    def close(self) -> None:
        """Close the client session."""
        self._session.close()

    def __enter__(self) -> "ISAPIClient":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()
