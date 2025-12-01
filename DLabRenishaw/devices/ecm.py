#!/usr/bin/python3
#
# Copyright (c) 2022 Renishaw plc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Connection class for the WiRE Externally Controlled Measurement.
"""

import requests
import json
import time
from requests.exceptions import ReadTimeout

class ECMException(Exception):
    """Exception raise on an error response from WiRE"""
    def __init__(self, message):
        super(ECMException, self).__init__(message)
        self.message = message


class ECMConnection():
    """
    Opens a connection to the remote WiRE system
    """

    def __init__(self, url):
        """
        Set the URL, request/response id, default debug mode and headers for the ecm connection
        """
        self.url = url
        self._id = 0
        self.debug = False
        self.rpctimeout = 1.0
        self.waitRetries = 3
        self.headers = {'content-type': 'application/json'}

    @property
    def id(self):
        """
        Each JSON-RPC call gets a unique identifier by calling this property.
        """
        id = self._id
        self._id += 1
        return id

    def call(self, methodName, **kwargs):
        """
        Generic JSON-RPC method calling for the ECM API.
        """
        result = None
        data = dict(jsonrpc="2.0", id=self.id, method=methodName, params=kwargs)
        if self.debug:
            print(data)
        try:
            res = requests.post(self.url, headers=self.headers, json=data, timeout=self.rpctimeout, proxies={'http': None})
            if self.debug:
                print(res.text)
        except ReadTimeout:
            print(f"Read timeout exception")
        except TimeoutError:
            print(f"Timed out requests call: {res}")
        if res.status_code == requests.codes['ok']:
            r = json.loads(res.text)
            if 'error' in r:
                raise ECMException(r['error']['message'])
            else:
                result = r['result']
        else:
            raise ECMException(res.text)
        return result

    def wait(self, handle, timeout=10000):
        """
        Wait for a specified measurement to complete with a timeout limit.
        If we timeout then the status result will not be "COMPLETE".
        """
        # Wait for the measurement status to change
        #time.sleep(0.250)
        status = ""
        pastStatus = ""
        consecutiveErrors = 0
        while status != "COMPLETE" and timeout > 0:
            try:
                status = self.call("Queue.GetMeasurementState", handle=handle)
                if pastStatus != status:
                    pastStatus=status
                    print("Measurement status is " + status)
            except ECMException as ex:
                print(ex)
                consecutiveErrors += 1
                if consecutiveErrors > self.waitRetries:
                    timeout = 0
                    status = f"ERRORS EXCEEDED {self.waitRetries} ALLOWED RETRIES"
            else:
                consecutiveErrors = 0
            time.sleep(0.250)
            timeout -= 250
        return status
