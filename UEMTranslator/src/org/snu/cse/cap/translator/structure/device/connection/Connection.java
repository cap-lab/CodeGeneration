package org.snu.cse.cap.translator.structure.device.connection;

import java.util.ArrayList;
import java.util.HashMap;

import org.snu.cse.cap.translator.Constants;
import org.snu.cse.cap.translator.structure.device.DeviceCommunicationType;

import hopes.cic.xml.NetworkType;

public abstract class Connection {
	protected NetworkType network;
	protected ProtocolType protocol;
	protected DeviceCommunicationType communication;
	protected String name;
	protected String role;
	protected Boolean usedEncryption;
	protected int encryptionListIndex;
	protected HashMap<String, ArrayList<Integer>> user;
	
	public Connection(String name, String role, NetworkType network, ProtocolType protocol) {
		this.name = name;
		this.role = role;
		this.network = network;
		this.protocol = protocol;
		this.usedEncryption = false;
		this.encryptionListIndex = Constants.INVALID_VALUE;

		userListInit();
	}

	public String getName() {
		return name;
	}
	
	public String getRole() {
		return role;
	}
	
	public void setName(String name) {
		this.name = name;
	}
	
	public void setRole(String role) {
		this.role = role;
	}

	public NetworkType getNetwork() {
		return network;
	}

	public ProtocolType getProtocol() {
		return protocol;
	}

	public Boolean getUsedEncryption() {
		return usedEncryption;
	}

	public void setUsedEncryption(Boolean usedEncryption) {
		this.usedEncryption = usedEncryption;
	}

	public int getEncryptionListIndex() {
		return encryptionListIndex;
	}

	public void setEncryptionListIndex(int index) {
		if (index != Constants.INVALID_VALUE) {
			this.setUsedEncryption(true);
		}
		encryptionListIndex = index;
	}
	
	public void putMulticastReceiver(Integer groupId) {
		user.get("multicastInput").add(groupId);
	}
	
	public ArrayList<Integer> getMulticastReceivers() {
		return user.get("multicastInput");
	}
	
	public void putMulticastSender(Integer groupId) {
		user.get("multicastOutput").add(groupId);
	}
	
	public ArrayList<Integer> getMulticastSenders() {
		return user.get("multicastOutput");
	}
	
	private void userListInit() {
		user = new HashMap<String, ArrayList<Integer>>();
		user.put("multicastInput", new ArrayList<Integer>());
		user.put("multicastOutput", new ArrayList<Integer>());
	}
}
