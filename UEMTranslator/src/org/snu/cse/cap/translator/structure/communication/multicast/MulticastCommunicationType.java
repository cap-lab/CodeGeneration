package org.snu.cse.cap.translator.structure.communication.multicast;

import java.util.ArrayList;

import org.snu.cse.cap.translator.structure.device.DeviceCommunicationType;

public enum MulticastCommunicationType {
	MULTICAST_COMMUNICATION_INVALID_TYPE,
	MULTICAST_COMMUNICATION_TYPE_SHARED_MEMORY,
	MULTICAST_COMMUNICATION_TYPE_UDP,
	;
	static public ArrayList<DeviceCommunicationType> getMulticastAvailableCommnicationTypeList()
	{
		ArrayList<DeviceCommunicationType> multicastAvailableConnectionTypeList = new ArrayList<DeviceCommunicationType>();
		multicastAvailableConnectionTypeList.add(DeviceCommunicationType.UDP);
		return multicastAvailableConnectionTypeList;
	}
	static public MulticastCommunicationType getMulticastCommunicationTypeByDeviceCommunicationType(DeviceCommunicationType deviceCommunicationType)
	{
		switch(deviceCommunicationType)
		{
		case UDP:
			return MULTICAST_COMMUNICATION_TYPE_UDP;
		default :
			return MULTICAST_COMMUNICATION_INVALID_TYPE;
		}
	}
}
