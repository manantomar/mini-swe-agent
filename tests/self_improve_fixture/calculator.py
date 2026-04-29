"""A buggy calculator module. The agent needs to fix the bugs."""


def add(a, b):
    return a + b


def subtract(a, b):
    return a - b


def multiply(a, b):
    return a * b


def divide(a, b):
    return a / b  # BUG: no zero division check


def average(numbers):
    return sum(numbers) / len(numbers)  # BUG: crashes on empty list
