package org.snu.cse.cap.translator.structure.device.connection;

import hopes.cic.xml.NetworkType;

public class ConstrainedSerialConnection extends SerialConnection {
	private int boardTXPinNumber;
	private int boardRXPinNumber; 

	public ConstrainedSerialConnection(String name, String role, NetworkType network, int boardTXPinNumber, int boardRXPinNumber) {
		super(name, role, network);
		this.boardTXPinNumber = boardTXPinNumber;
		this.boardRXPinNumber = boardRXPinNumber;
	}

	public int getBoardTXPinNumber() {
		return boardTXPinNumber;
	}

	public int getBoardRXPinNumber() {
		return boardRXPinNumber;
	}

}
