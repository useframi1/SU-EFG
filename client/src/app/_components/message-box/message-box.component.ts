import { Component, input } from '@angular/core';
import { Message } from '../../_models/message';

@Component({
  selector: 'app-message-box',
  standalone: true,
  imports: [],
  templateUrl: './message-box.component.html',
  styleUrl: './message-box.component.css',
})
export class MessageBoxComponent {
  conversation = input.required<Message[]>();
}
