package org.snu.cse.cap.translator.structure.device.connection;

import java.util.ArrayList;
import java.util.HashMap;

import hopes.cic.xml.NetworkType;

public abstract class Connection {
	protected NetworkType network;
	protected ProtocolType protocol;
	protected String name;
	protected String role;
	protected HashMap<String, ArrayList<Integer>> user;
	
	public Connection(String name, String role, NetworkType network, ProtocolType protocol) 
	{
		this.name = name;
		this.role = role;
		this.network = network;
		this.protocol = protocol;
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
